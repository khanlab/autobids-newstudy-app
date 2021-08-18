from autobidsportal import app, db
from autobidsportal.models import User, Answer, Task, Notification, Cfmm2tar, Tar2bids
from autobidsportal.dcm4cheutils import Dcm4cheUtils, gen_utils, Cfmm2tarError, Tar2bidsError
from rq import get_current_job
from datetime import datetime
import time
import os

def _set_task_progress(progress, error):
    """ Checks the progress of a Task and updates the task in the "task" table in the database accordingly.

    """
    job = get_current_job()
    if job:
        job.meta['progress'] = progress
        job.save_meta()
        task = Task.query.get(job.get_id())
        task.user.add_notification('task_progress', {'task_id': job.get_id(), 'progress': progress})
        if progress == 50:
            task.complete = True
            task.success = False
            task.error = error
            task.end_time = datetime.utcnow()
        elif progress == 100:
            task.complete = True
            task.success = True
            task.error = None
            task.end_time = datetime.utcnow()
        db.session.commit()

def get_info_from_cfmm2tar(user_id, second_last_pressed_button_id, button_id):
    """ Adds each result in the new_results array to the "cfmm2tar" table in the database.

    """
    _set_task_progress(0, "None")
    user = User.query.get(user_id)
    submitter_answer = db.session.query(Answer).filter(Answer.submitter_id==button_id)[0]
    if submitter_answer.principal_other != '':
        study_info = f"{submitter_answer.principal_other}^{submitter_answer.project_name}"
    else:
        study_info = f"{submitter_answer.principal}^{submitter_answer.project_name}"
    prefix = app.config["CFMM2TAR_DOWNLOAD_DIR"]
    data = "%s/%s/%s" % (prefix, user.id, datetime.utcnow().strftime("%Y%m%d%H%M"))
    try:
        new_results = get_new_cfmm2tar_results(study_info=study_info, data=data, button_id=button_id)
        for r in new_results:        
            day = r[0].rsplit('/', 3)[3].rsplit('_', 5)[0].rsplit('_', 5)[5]
            month = r[0].rsplit('/', 3)[3].rsplit('_', 5)[0].rsplit('_', 5)[4]
            year = r[0].rsplit('/', 3)[3].rsplit('_', 5)[0].rsplit('_', 5)[3]
            date = datetime(int(year), int(month), int(day))
            uid_file = open(r[1], "r")
            uid = (uid_file.read()) 
            cfmm2tar = Cfmm2tar(user_id=user_id, tar_file=r[0], uid_file=uid, task_button_id=button_id, date=date)
            db.session.add(cfmm2tar)
        db.session.commit()
        time.sleep(10)
        _set_task_progress(100, "None")  
        
    except Cfmm2tarError as err:
        _set_task_progress(50, err.__cause__.stderr)
        if 'cfmm2tar_intermediate_dicoms' in os.listdir(data):
            os.listdir(data)
        return err

def get_new_cfmm2tar_results(study_info, data, button_id):
    """ Runs cfmm2tar and appends any new results to the new_results array.

    """
    cfmm2tar_result = gen_utils().run_cfmm2tar(out_dir=data, project=study_info)
    cfmm2tar_results_in_db = Cfmm2tar.query.filter_by(task_button_id = button_id).all()
    new_results = []
    already_there = []
    if cfmm2tar_result == []:
        err = "Invalid Principal or Project Name"
        _set_task_progress(50, err)
    else:
        for result in cfmm2tar_result:
            if cfmm2tar_results_in_db != []:
                for db_result in cfmm2tar_results_in_db:
                    if result[0].rsplit('/', 3)[3]==db_result.tar_file.rsplit('/', 3)[3]:
                        already_there.append(result)
                    else:
                        if result not in already_there and result not in new_results:
                            new_results.append(result)
            else:
                new_results.append(result)

    if already_there != []:
    	for new in new_results:
          if new in already_there:
            new_results.remove(new)

    return(new_results)

def get_info_from_tar2bids(user_id, button_id, tar_file_id):
    """ Runs tar2bids and adds the tar2bids_result to the "tar2bids" table in the database.

    """
    _set_task_progress(0, "None")
    user = User.query.get(user_id)
    selected_heuristic = user.selected_heuristic
    submitter_answer = db.session.query(Answer).filter(Answer.submitter_id==button_id)[0]
    if submitter_answer.principal_other != '':
        study_info = f"{submitter_answer.principal_other}^{submitter_answer.project_name}"
    else:
        study_info = f"{submitter_answer.principal}^{submitter_answer.project_name}"
    tar_file = Cfmm2tar.query.filter_by(id = tar_file_id)[0].tar_file
    prefix = app.config["TAR2BIDS_DOWNLOAD_DIR"]
    data = "%s/%s/%s" % (prefix, study_info, button_id)
    if os.path.isdir(data) != True:
        os.makedirs(data)
    try:
        tar2bids_results = gen_utils().run_tar2bids(output_dir=data, tar_files=[tar_file], heuristic=selected_heuristic)
        tar2bids = Tar2bids(user_id=user_id, tar_file_id=tar_file_id, task_button_id=button_id, tar_file=tar_file, bids_file=tar2bids_results, heuristic=selected_heuristic)
        db.session.add(tar2bids)
        db.session.commit()
        time.sleep(10)
        _set_task_progress(100, "None")
    except Tar2bidsError as err:
        _set_task_progress(50, err.__cause__.stderr)
        return err