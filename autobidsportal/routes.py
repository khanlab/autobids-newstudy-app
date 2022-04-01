"""All routes in the portal are defined here."""

from datetime import datetime
from json import JSONEncoder, loads, dumps
from pathlib import Path
from smtplib import SMTPAuthenticationError
import shutil

from flask import (
    current_app,
    Blueprint,
    render_template,
    flash,
    redirect,
    url_for,
    request,
    abort,
)
from flask_login import login_user, logout_user, current_user, login_required
from flask_mail import Mail, Message
import flask_excel as excel

from werkzeug.urls import url_parse
from autobidsportal.models import (
    db,
    User,
    Study,
    Principal,
    Task,
    Cfmm2tarOutput,
    Tar2bidsOutput,
    ExplicitPatient,
)
from autobidsportal.dcm4cheutils import (
    Dcm4cheError,
)
from autobidsportal.forms import (
    LoginForm,
    BidsForm,
    RegistrationForm,
    AccessForm,
    RemoveAccessForm,
    StudyConfigForm,
    Tar2bidsRunForm,
    ExcludeScansForm,
    IncludeScansForm,
    DEFAULT_HEURISTICS,
)
from autobidsportal.filesystem import gen_dir_dict
from autobidsportal.dicom import get_study_records

portal_blueprint = Blueprint(
    "portal_blueprint", __name__, template_folder="templates"
)

mail = Mail()


@portal_blueprint.route("/", methods=["GET", "POST"])
@portal_blueprint.route("/index", methods=["GET", "POST"])
def index():
    """Provides a survey form for users to fill out.

    If the Principal table is not empty, the principal names are added to
    the principal dropdown in the form. Submitter information and their
    answer is added to the database.
    """
    form = BidsForm()
    principal_names = [
        (p.principal_name, p.principal_name)
        for p in db.session.query(Principal).all()
    ]
    form.principal.choices = principal_names
    form.principal.choices.insert(0, ("Other", "Other"))

    if form.validate_on_submit():
        principal = (
            form.principal_other.data
            if form.principal.data == "Other"
            else form.principal.data
        )

        if (
            len(
                Study.query.filter(
                    Study.principal == principal,
                    Study.project_name == form.project_name.data,
                ).all()
            )
            > 0
        ):
            flash("That study already exists.")
            abort(400)

        if form.retrospective_data.data:
            retrospective_start = form.retrospective_start.data
            retrospective_end = form.retrospective_end.data
        else:
            retrospective_start = None
            retrospective_end = None

        dataset_name = (
            form.dataset_name.data
            if form.dataset_name.data != ""
            else form.project_name.data
        )

        answer = Study(
            submitter_name=form.name.data,
            submitter_email=form.email.data,
            status=form.status.data,
            scanner=form.scanner.data,
            scan_number=form.scan_number.data,
            study_type=form.study_type.data,
            familiarity_bids=form.familiarity_bids.data,
            familiarity_bidsapp=form.familiarity_bidsapp.data,
            familiarity_python=form.familiarity_python.data,
            familiarity_linux=form.familiarity_linux.data,
            familiarity_bash=form.familiarity_bash.data,
            familiarity_hpc=form.familiarity_hpc.data,
            familiarity_openneuro=form.familiarity_openneuro.data,
            familiarity_cbrain=form.familiarity_cbrain.data,
            principal=principal,
            project_name=form.project_name.data,
            dataset_name=dataset_name,
            sample=form.sample.data,
            retrospective_data=form.retrospective_data.data,
            retrospective_start=retrospective_start,
            retrospective_end=retrospective_end,
            consent=form.consent.data,
            comment=form.comment.data,
            tar2bids_img=current_app.config["TAR2BIDS_DEFAULT_IMAGE"],
        )

        db.session.add(answer)
        db.session.commit()

        current_app.logger.info("Study %i successfully submitted.", answer.id)

        flash("Thanks, the survey has been submitted!")

        if current_app.config["MAIL_ENABLED"]:
            subject = (
                f"A new request has been submitted by {form.name.data}"
                + f" {form.email.data}"
            )
            sender = current_app.config["MAIL_USERNAME"]
            recipients = current_app.config["MAIL_RECIPIENTS"]

            msg = Message(
                subject=subject,
                body="A new request has been submitted. Please login to "
                + "see the submitter's response",
                sender=sender,
                recipients=recipients.split(),
            )
            mail.send(msg)

        return redirect(url_for("portal_blueprint.index"))
    return render_template("survey.html", form=form)


@portal_blueprint.route("/login", methods=["GET", "POST"])
def login():
    """Redirects user to login if they their email and password is valid."""
    if current_user.is_authenticated:
        return redirect(url_for("portal_blueprint.index"))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user is None or not user.check_password(form.password.data):
            flash("Invalid email or password")
            return redirect(url_for("portal_blueprint.login"))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get("next")
        if not next_page or url_parse(next_page).netloc != "":
            next_page = url_for("portal_blueprint.index")
        return redirect(next_page)
    return render_template("login.html", title="Sign In", form=form)


@portal_blueprint.route("/register", methods=["GET", "POST"])
def register():
    """Validates that provided email and password are valid.

    After the user is registered, they are redirected to login.
    """
    if current_user.is_authenticated:
        return redirect(url_for("portal_blueprint.index"))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(email=form.email.data, admin=False)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        current_app.logger.info("New user %i registered.", user.id)
        flash("Congratulations, you are now a registered user!")
        return redirect(url_for("portal_blueprint.login"))
    return render_template("register.html", title="Register", form=form)


@portal_blueprint.route("/admin", methods=["GET", "POST"])
@login_required
def user_list():
    """Obtains all the registered users from the database."""
    if not current_user.admin:
        abort(404)
    users = User.query.all()
    return render_template("admin.html", title="Administration", users=users)


@portal_blueprint.route("/admin/update_heuristics", methods=["POST"])
@login_required
def update_heuristics():
    """Launch an update heuristics task."""
    if not current_user.admin:
        abort(404)

    current_user.launch_task(
        "update_heuristics", "Manually triggered heuristic update"
    )
    current_app.logger.info("Update heuristic task launched.")
    flash("Currently updating heuristics... Give it a minute or two.")
    return user_list()


@portal_blueprint.route("/admin/<int:user_id>", methods=["GET", "POST"])
@login_required
def admin(user_id):
    """Exposes information about a specific registered user.

    A GET request shows a page with information about the user.

    A POST request alters some aspect of the user, then shows the same page.

    Parameters
    ----------
    user_id : int
        ID of the user to expose.
    """
    if not current_user.admin:
        abort(404)
    form = AccessForm()
    removal_form = RemoveAccessForm()
    all_studies = db.session.query(Study).all()
    form.choices.choices = [
        (study.id, f"{study.principal}^{study.project_name}")
        for study in all_studies
    ]
    removal_form.choices_to_remove.choices = form.choices.choices
    user = User.query.get(user_id)
    if request.method == "POST":
        if "admin" in request.form:
            make_admin = request.form["admin"].lower() == "true"
            if make_admin:
                user.admin = True
            else:
                user.admin = False
            db.session.commit()
            current_app.logger.info(
                "Changed user %i's admin status to %s", user.id, user.admin
            )
        if form.validate_on_submit():
            for study_id in form.choices.data:
                print(study_id)
                study = Study.query.get(study_id)
                current_app.logger.info(
                    "Added user %i to study %i.", user.id, study.id
                )
                if user not in study.users_authorized:
                    study.users_authorized.append(user)
            db.session.commit()
        if removal_form.validate_on_submit():
            for study_id in removal_form.choices_to_remove.data:
                study = Study.query.get(study_id)
                if user in study.users_authorized:
                    study.users_authorized.remove(user)
                    current_app.logger.info(
                        "Removed user %i from study %i", user.id, study.id
                    )
            db.session.commit()

    return render_template(
        "administration.html",
        title="Administration",
        form=form,
        removal_form=removal_form,
        user=user,
    )


@portal_blueprint.route("/results", methods=["GET"])
@login_required
def results():
    """Get responses and the date and time the current user last logged in."""
    last = current_user.last_seen

    if current_user.admin:
        studies = Study.query.all()
    else:
        studies = current_user.studies

    studies = sorted(
        studies,
        key=lambda x: x.submission_date,
        reverse=True,
    )

    return render_template(
        "results.html", title="Responses", answers=studies, last=last
    )


@portal_blueprint.route("/results/<int:study_id>", methods=["GET"])
@login_required
def answer_info(study_id):
    """Obtains complete survey response based on the submission id"""
    study = Study.query.get_or_404(study_id)
    if (not current_user.admin) and (
        current_user not in study.users_authorized
    ):
        abort(404)
    cfmm2tar_tasks = Task.query.filter_by(
        study_id=study_id, name="get_info_from_cfmm2tar"
    ).all()
    cfmm2tar_files = study.cfmm2tar_outputs
    cfmm2tar_file_names = [
        Path(cfmm2tar_file.tar_file).name for cfmm2tar_file in cfmm2tar_files
    ]
    tar2bids_tasks = Task.query.filter_by(
        study_id=study_id, name="get_info_from_tar2bids"
    ).all()
    tar2bids_files = study.tar2bids_outputs
    tar2bids_path = (
        Path(current_app.config["TAR2BIDS_DOWNLOAD_DIR"])
        / str(study.id)
        / (
            study.dataset_name
            if study.dataset_name not in [None, ""]
            else study.project_name
        )
    )

    bids_dict = (
        gen_dir_dict(tar2bids_path)
        if tar2bids_path.exists()
        else {"files": [], "dirs": []}
    )
    json_filetree = JSONEncoder().encode(bids_dict)

    form = Tar2bidsRunForm()
    form.tar_files.choices = [
        (tar_file.id, "Yes") for tar_file in cfmm2tar_files
    ]
    form.tar_files.default = []
    form.process()
    return render_template(
        "answer_info.html",
        title="Response",
        submitter_answer=study,
        cfmm2tar_tasks=cfmm2tar_tasks,
        button_id=study_id,
        form_data=zip(form.tar_files, cfmm2tar_file_names, cfmm2tar_files),
        tar2bids_tasks=tar2bids_tasks,
        tar2bids_files=tar2bids_files,
        json_filetree=json_filetree,
    )


@portal_blueprint.route(
    "/results/<int:study_id>/demographics", methods=["GET"]
)
def study_demographics(study_id):
    """Render page with information about a study's submitter."""
    study = Study.query.get_or_404(study_id)
    if (not current_user.admin) and (
        current_user not in study.users_authorized
    ):
        abort(404)
    return render_template("study_demographics.html", study=study)


@portal_blueprint.route(
    "/results/<int:study_id>/config", methods=["GET", "POST"]
)
@login_required  # pylint: disable=too-many-statements,too-many-branches
def study_config(study_id):
    """Page to display and edit study config."""
    # pylint: disable=too-many-statements,too-many-branches
    study = Study.query.get_or_404(study_id)
    if (not current_user.admin) and (
        current_user not in study.users_authorized
    ):
        abort(404)

    form = StudyConfigForm()
    if request.method == "POST":
        study.principal = form.pi_name.data
        study.project_name = form.project_name.data
        study.dataset_name = form.dataset_name.data
        if form.example_date.data:
            study.sample = form.example_date.data
        else:
            study.sample = None
        if form.retrospective_data.data:
            study.retrospective_data = True
            study.retrospective_start = form.retrospective_start.data
            study.retrospective_end = form.retrospective_end.data
        else:
            study.retrospective_data = False
            study.retrospective_start = None
            study.retrospective_end = None
        study.heuristic = form.heuristic.data
        study.tar2bids_img = form.tar2bids_img.data
        study.subj_expr = form.subj_expr.data
        study.patient_str = form.patient_str.data
        study.patient_name_re = form.patient_re.data
        for explicit_patient in study.explicit_patients:
            if explicit_patient.included and (
                explicit_patient.study_instance_uid
                not in form.included_patients.data
            ):
                db.session.delete(explicit_patient)
            elif (not explicit_patient.included) and (
                explicit_patient.study_instance_uid
                not in form.excluded_patients.data
            ):
                db.session.delete(explicit_patient)
        if form.newly_excluded.data != "":
            db.session.add(
                ExplicitPatient(
                    study_id=study.id,
                    study_instance_uid=form.newly_excluded.data,
                    included=False,
                )
            )
        if form.newly_included.data != "":
            db.session.add(
                ExplicitPatient(
                    study_id=study.id,
                    study_instance_uid=form.newly_included.data,
                    included=True,
                )
            )
        study.users_authorized = [
            User.query.get(id) for id in form.users_authorized.data
        ]
        current_app.logger.info("Updated study %i config", study.id)
        db.session.commit()

    available_heuristics = sorted(
        [
            (str(heuristic_path), f"{heuristic_path.name} (git)")
            for heuristic_path in (
                Path(current_app.config["HEURISTIC_REPO_PATH"])
                / current_app.config["HEURISTIC_DIR_PATH"]
            ).iterdir()
        ]
        + [
            (heuristic, f"{heuristic} (container)")
            for heuristic in DEFAULT_HEURISTICS
        ],
        key=lambda option: option[1].lower(),
    )
    available_tar2bids_imgs = sorted(
        [
            (path.name, path.name)
            for path in Path(
                current_app.config["TAR2BIDS_IMAGE_DIR"]
            ).iterdir()
        ],
        key=lambda option: option[1].lower(),
    )

    principal_names = [
        p.principal_name for p in db.session.query(Principal).all()
    ]
    if study.principal not in principal_names:
        principal_names.insert(0, study.principal)
    form.pi_name.choices = principal_names
    form.pi_name.default = study.principal
    form.project_name.default = study.project_name
    if study.dataset_name is not None:
        form.dataset_name.default = study.dataset_name
    if study.sample is not None:
        form.example_date.default = study.sample
    form.retrospective_data.default = study.retrospective_data
    if study.retrospective_data:
        form.retrospective_start.default = study.retrospective_start
        form.retrospective_end.default = study.retrospective_end
    form.heuristic.choices = available_heuristics
    if study.heuristic is None:
        form.heuristic.default = "cfmm_base.py"
    else:
        form.heuristic.default = study.heuristic
    form.tar2bids_img.choices = available_tar2bids_imgs
    form.tar2bids_img.default = study.tar2bids_img
    if study.subj_expr is None:
        form.subj_expr.default = "*_{subject}"
    else:
        form.subj_expr.default = study.subj_expr
    form.patient_str.default = study.patient_str
    if study.patient_name_re is None:
        form.patient_re.default = ".*"
    else:
        form.patient_re.default = study.patient_name_re
    form.excluded_patients.choices = [
        (
            patient.study_instance_uid,
            (
                f"Patient Name: {patient.patient_name}, "
                f"Study ID: {patient.dicom_study_id}"
            ),
        )
        for patient in study.explicit_patients
        if not patient.included
    ]
    form.excluded_patients.default = [
        patient.study_instance_uid
        for patient in study.explicit_patients
        if not patient.included
    ]
    form.newly_excluded.default = ""
    form.included_patients.choices = [
        (
            patient.study_instance_uid,
            (
                f"Patient Name: {patient.patient_name}, "
                f"Study ID: {patient.dicom_study_id}"
            ),
        )
        for patient in study.explicit_patients
        if patient.included
    ]
    form.included_patients.default = [
        patient.study_instance_uid
        for patient in study.explicit_patients
        if patient.included
    ]
    form.newly_included.default = ""
    form.users_authorized.choices = [
        (user.id, user.email) for user in User.query.all()
    ]
    form.users_authorized.default = [
        user.id for user in study.users_authorized
    ]

    form.process()

    return render_template("study_config.html", form=form, study=study)


@portal_blueprint.route("/results/<int:study_id>/cfmm2tar", methods=["POST"])
@login_required
def run_cfmm2tar(study_id):
    """Launch cfmm2tar task and refresh answer_info.html"""
    study = Study.query.get_or_404(study_id)
    if (not current_user.admin) and (
        current_user not in study.users_authorized
    ):
        abort(404)
    if (
        len(
            Task.query.filter_by(
                study_id=study_id,
                name="get_info_from_cfmm2tar",
                complete=False,
            ).all()
        )
        > 0
    ):
        flash("An Cfmm2tar run is currently in progress")
    else:
        current_user.launch_task(
            "get_info_from_cfmm2tar",
            f"cfmm2tar for study {study_id}",
            study_id,
        )
        current_app.logger.info("Launched cfmm2tar for study %i", study_id)
        db.session.commit()
    if current_app.config["MAIL_ENABLED"]:
        subject = (
            f"A Cfmm2tar run for {study.prinicipal}^{study.project_name} "
            f"has been submitted by {current_user.email}."
        )
        body = (
            f"A Cfmm2tar run for {study.principal}^{study.project_name} "
            "has been submitted."
        )
        sender = current_app.config["MAIL_USERNAME"]
        recipients = current_app.config["MAIL_RECIPIENTS"]

        msg = Message(
            subject=subject,
            body=body,
            sender=sender,
            recipients=recipients.split(),
        )
        try:
            mail.send(msg)
        except SMTPAuthenticationError as err:
            print(err)

    return answer_info(study_id)


@portal_blueprint.route(
    "/results/<int:study_id>/cfmm2tar/<int:cfmm2tar_id>/delete",
    methods=["GET"],
)
@portal_blueprint.route(
    "/results/<int:study_id>/cfmm2tar/<int:cfmm2tar_id>", methods=["DELETE"]
)
@login_required
def delete_cfmm2tar(study_id, cfmm2tar_id):
    """Delete a single tar file."""
    study = Study.query.get_or_404(study_id)
    if (not current_user.admin) and (
        current_user not in study.users_authorized
    ):
        abort(404)
    cfmm2tar_output = Cfmm2tarOutput.query.get(cfmm2tar_id)
    if (cfmm2tar_output is not None) and (
        cfmm2tar_output.study_id == study_id
    ):
        cfmm2tar_file = Path(cfmm2tar_output.tar_file).resolve()
        cfmm2tar_dir = cfmm2tar_file.parent
        cfmm2tar_file.unlink()
        if len(list(cfmm2tar_dir.iterdir())) == 0:
            cfmm2tar_dir.rmdir()
        db.session.delete(cfmm2tar_output)
        current_app.logger.info(
            "Deleted tar file %s for study %i", cfmm2tar_file, study_id
        )
        db.session.commit()

    return answer_info(study_id)


@portal_blueprint.route(
    "/results/<int:study_id>/tar2bids/delete",
    methods=["GET"],
)
@portal_blueprint.route("/results/<int:study_id>/tar2bids", methods=["DELETE"])
@login_required
def delete_tar2bids(study_id):
    """Delete a study's BIDS directory."""
    study = Study.query.get_or_404(study_id)
    if (not current_user.admin) and (
        current_user not in study.users_authorized
    ):
        abort(404)
    tar2bids_outputs = Tar2bidsOutput.query.filter_by(study_id=study_id).all()
    if len(tar2bids_outputs) > 0:
        for tar2bids_output in tar2bids_outputs:
            tar2bids_path = Path(tar2bids_output.bids_dir).resolve()
            if tar2bids_path.exists():
                shutil.rmtree(str(tar2bids_path))
            db.session.delete(tar2bids_output)
            current_app.logger.info(
                "Deleted BIDS dir %s for study %i", tar2bids_path, study_id
            )
            db.session.commit()

    return answer_info(study_id)


@portal_blueprint.route("/results/<int:study_id>/tar2bids", methods=["POST"])
@login_required
def run_tar2bids(study_id):
    """Launch tar2bids task and refresh answer_info.html"""
    study = Study.query.get_or_404(study_id)
    if (not current_user.admin) and (
        current_user not in study.users_authorized
    ):
        abort(404)
    form = Tar2bidsRunForm()
    tar_files = [
        Cfmm2tarOutput.query.get_or_404(tar_file_id)
        for tar_file_id in form.tar_files.data
    ]
    tar_files = [
        tar_file for tar_file in tar_files if tar_file.study_id == study_id
    ]

    if (
        len(
            Task.query.filter_by(
                study_id=study_id,
                name="get_info_from_tar2bids",
                complete=False,
            ).all()
        )
        > 0
    ):
        flash("An tar2bids run is currently in progress")
    else:
        current_user.launch_task(
            "get_info_from_tar2bids",
            f"tar2bids for study {study_id}",
            study_id,
            [tar_file.id for tar_file in tar_files],
        )
        current_app.logger.info(
            "Launched tar2bids for study %i with files %s",
            study_id,
            [tar_file.tar_file for tar_file in tar_files],
        )
        db.session.commit()

        if current_app.config["MAIL_ENABLED"]:
            subject = (
                f"A Tar2bids run for {study.principal}^{study.project_name} "
                f"has been submitted by {current_user.email}"
            )
            body = (
                f"A Tar2bids run for {study.principal}^{study.project_name} "
                "has been submitted."
            )
            sender = current_app.config["MAIL_USERNAME"]
            recipients = current_app.config["MAIL_RECIPIENTS"]

            try:
                mail.send(
                    Message(
                        subject=subject,
                        body=body,
                        sender=sender,
                        recipients=recipients.split(),
                    )
                )
            except SMTPAuthenticationError as err:
                print(err)

    return answer_info(study_id)


@portal_blueprint.route("/results/download", methods=["GET"])
@login_required
def download():
    """Downloads csv containing all the survey response"""
    if current_user.admin:
        response_list = Study.query.all()
    else:
        response_list = current_user.studies
    file_name = "Response_report"

    csv_list = [
        [file_name],
        [
            "Submitter Name",
            "Submitter Email",
            "Status",
            "Scanner",
            "Number of Scans",
            "Study Type",
            "Bids Familiarity",
            "Bids App Familiarity",
            "Python Familiarity",
            "Linux Familiarity",
            "Bash Familiarity",
            "HPC Familiarity",
            "OPENNEURO Familiarity",
            "CBRAIN Familiarity",
            "Principal",
            "Project Name",
            "Dataset Name",
            "Sample Date",
            "Retrospective Data",
            "Retrospective Data Start Date",
            "Retrospective Data End Date",
            "Consent",
            "Comment",
        ],
    ]

    def update_scanner(scanner):
        return "3T" if scanner == "type1" else "7T"

    def update_familiarity(familiarity):
        if familiarity == "1":
            new_familiarity = "Not familiar at all"
        elif familiarity == "2":
            new_familiarity = "Have heard of it"
        elif familiarity == "3":
            new_familiarity = "Have used it before"
        elif familiarity == "4":
            new_familiarity = "Used it regularly"
        elif familiarity == "5":
            new_familiarity = "I consider myself an expert"
        return new_familiarity

    def update_date(date):
        return date.date() if date is not None else date

    def update_bool(bool_str):
        return "Yes" if bool_str == "1" else "No"

    for response in response_list:
        csv_list.append(
            [
                response.submitter_name,
                response.submitter_email,
                response.status.capitalize(),
                update_scanner(response.scanner),
                response.scan_number,
                update_bool(response.study_type),
                update_familiarity(response.familiarity_bids),
                update_familiarity(response.familiarity_bidsapp),
                update_familiarity(response.familiarity_python),
                update_familiarity(response.familiarity_linux),
                update_familiarity(response.familiarity_bash),
                update_familiarity(response.familiarity_hpc),
                update_familiarity(response.familiarity_openneuro),
                update_familiarity(response.familiarity_cbrain),
                response.principal,
                response.project_name,
                response.dataset_name,
                update_date(response.sample),
                update_bool(response.retrospective_data),
                update_date(response.retrospective_start),
                update_date(response.retrospective_end),
                update_bool(response.consent),
                response.comment,
            ]
        )
    return excel.make_response_from_array(csv_list, "csv", file_name=file_name)


@portal_blueprint.route("results/<int:study_id>/exclusions", methods=["POST"])
@login_required
def update_exclusions(study_id):
    """Updates the excluded UIDs for a study."""
    study = Study.query.get_or_404(study_id)
    if (not current_user.admin) and (
        current_user not in study.users_authorized
    ):
        abort(404)

    form_exclude = ExcludeScansForm()
    for val_json in form_exclude.choices_to_exclude.data:
        val = loads(loads(val_json))
        old_uid = ExplicitPatient.query.filter_by(
            study_instance_uid=val["StudyInstanceUID"]
        ).one_or_none()
        if old_uid is not None:
            db.session.delete(old_uid)

        excluded_uid = ExplicitPatient(
            study_id=study.id,
            study_instance_uid=val["StudyInstanceUID"],
            patient_name=val["PatientName"],
            dicom_study_id=val["StudyID"],
            included=False,
        )
        db.session.add(excluded_uid)
        db.session.commit()
    form_include = IncludeScansForm()
    for val_json in form_include.choices_to_include.data:
        val = loads(loads(val_json))
        old_uid = ExplicitPatient.query.filter_by(
            study_instance_uid=val["StudyInstanceUID"]
        ).one_or_none()
        if old_uid is not None:
            break
        included_uid = ExplicitPatient(
            study_id=study.id,
            study_instance_uid=val["StudyInstanceUID"],
            patient_name=val["PatientName"],
            dicom_study_id=val["StudyID"],
            included=True,
        )
        db.session.add(included_uid)
        db.session.commit()

    return dicom_verify(study_id, "description")


@portal_blueprint.route(
    "/results/<int:study_id>/dicom/<string:method>", methods=["GET"]
)
@login_required
def dicom_verify(study_id, method):
    """Gets all DICOM results for a specific study."""
    study = Study.query.get_or_404(study_id)
    if (not current_user.admin) and (
        current_user not in study.users_authorized
    ):
        abort(404)
    study_info = f"{study.principal}^{study.project_name}"
    if method.lower() == "both":
        if study.sample is None:
            abort(404)
        description = study_info
        date = study.sample.date()
    elif method.lower() == "date":
        if study.sample is None:
            abort(404)
        description = None
        date = study.sample.date()
    elif method.lower() == "description":
        description = study_info
        date = None
    else:
        abort(404)

    try:
        responses = get_study_records(
            study, date=date, description=description
        )
    except Dcm4cheError as err:
        err_cause = err.__cause__.stderr
        current_app.logger.warning(
            "Failed to get DICOM info for study %i: %s", study_id, err
        )
        return render_template(
            "dicom_error.html",
            err=err,
            err_cause=err_cause,
            title="DICOM Result Not Found",
        )

    sorted_responses = sorted(
        responses,
        key=lambda attr_dict: f'{attr_dict["PatientName"]}',
    )
    form_exclude = ExcludeScansForm()
    form_exclude.choices_to_exclude.choices = [
        (
            dumps(
                {
                    "StudyInstanceUID": response["StudyInstanceUID"],
                    "PatientName": response["PatientName"],
                    "StudyID": response["StudyID"],
                }
            ),
            "Exclude",
        )
        for response in sorted_responses
    ]
    form_include = IncludeScansForm()
    form_include.choices_to_include.choices = [
        (
            dumps(
                {
                    "StudyInstanceUID": response["StudyInstanceUID"],
                    "PatientName": response["PatientName"],
                    "StudyID": response["StudyID"],
                }
            ),
            "Include",
        )
        for response in responses
    ]

    return render_template(
        "dicom.html",
        title="Dicom Result",
        dicom_response=sorted_responses,
        submitter_answer=study,
        form_exclude=form_exclude,
        form_include=form_include,
    )


@portal_blueprint.route("/logout", methods=["GET", "POST"])
def logout():
    """Logs out current user."""
    if current_user.is_authenticated:
        # pylint doesn't like werkzeug proxies
        # pylint: disable=assigning-non-slot
        current_user.last_seen = datetime.utcnow()
        # pylint: enable=assigning-non-slot
        db.session.commit()
    logout_user()
    return redirect(url_for("portal_blueprint.index"))
