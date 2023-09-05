"""Utilities to handle tasks put on the queue."""
from __future__ import annotations

import pathlib
import re
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from os import PathLike
from shutil import copy2, rmtree
from typing import Protocol
from zipfile import ZipFile

from bids import BIDSLayout
from datalad.support.gitrepo import GitRepo

from autobidsportal.app import create_app
from autobidsportal.apptainer import apptainer_exec
from autobidsportal.bids import merge_datasets
from autobidsportal.datalad import (
    RiaDataset,
    archive_dataset,
    ensure_dataset_exists,
    finalize_dataset_changes,
    get_all_dataset_content,
    get_tar_file_from_dataset,
)
from autobidsportal.dateutils import TIME_ZONE
from autobidsportal.dcm4cheutils import (
    Cfmm2tarArgs,
    Cfmm2tarError,
    Cfmm2tarTimeoutError,
    Tar2bidsArgs,
    Tar2bidsError,
    gen_utils,
)
from autobidsportal.dicom import get_study_records
from autobidsportal.email import send_email
from autobidsportal.filesystem import gen_dir_dict, render_dir_dict
from autobidsportal.models import (
    Cfmm2tarOutput,
    DataladDataset,
    DatasetArchive,
    DatasetType,
    Study,
    Tar2bidsOutput,
    Task,
    User,
    db,
)
from autobidsportal.ssh import copy_file, make_remote_dir

app = create_app()
app.app_context().push()

COMPLETION_PROGRESS = 100
MAX_CFMM2TAR_ATTEMPTS = 5


class Loggable(Protocol):
    """Something that can store logs."""

    def log(self, msg: str) -> None:
        """Record a log message."""
        ...


class NullLogger:
    """Logger that discards provided messages."""

    def log(self, msg: str) -> None:
        """Discard the provided message."""


@dataclass
class TaskLogger:
    """Logger that logs messages to a Task."""

    task_id: str

    @cached_property
    def task(self) -> Task:
        """Get the corresponding task."""
        return Task.query.get(self.task_id)

    def log(self, msg: str) -> None:
        """Append message to a task's persistent log."""
        existing_log = self.task.log or ""
        self.task.log = f"{existing_log}\n{msg}"
        db.session.commit()


def log_success(job):
    """Reflect a job success in the DB."""
    task = Task.query.get(job.id)
    task.complete = True
    task.success = True
    task.end_time = datetime.now(tz=TIME_ZONE)
    db.session.commit()


def log_failure(job, error: str):
    """Reflect a job failure in the DB."""
    task = Task.query.get(job.id)
    task.complete = True
    task.success = False
    task.end_time = datetime.now(tz=TIME_ZONE)
    task.error = str(error)
    db.session.commit()


def run_cfmm2tar_with_retries(
    out_dir: str,
    study_instance_uid: str,
) -> tuple[list[list[str]], str]:
    """Run cfmm2tar, retrying multiple times if it times out.

    Parameters
    ----------
    out_dir
        Directory to which to download tar files.
    study_instance_uid
        StudyInstanceUid to download

    Raises
    ------
    Cfmm2tarTimeoutError
        If cfmm2tar times out too many times.
    """
    cfmm2tar_result, log = [], ""
    for attempt in range(1, 6):
        try:
            cfmm2tar_result, log = gen_utils().run_cfmm2tar(
                Cfmm2tarArgs(
                    out_dir=out_dir,
                    study_instance_uid=study_instance_uid,
                ),
            )
        except Cfmm2tarTimeoutError:
            if attempt < MAX_CFMM2TAR_ATTEMPTS:
                app.logger.warning(
                    "cfmm2tar timeout after %i attempt(s) (target %s).",
                    attempt,
                    study_instance_uid,
                )
                continue
            raise
        break
    return cfmm2tar_result, log


def record_cfmm2tar(tar_file, uid, study_id, attached_tar_file=None):
    """Parse cfmm2tar output files and record them in the db.

    Parameters
    ----------
    tar_file : str
        Name of the downloaded tar file.
    uid : str
        StudyInstanceUID of the tar file..
    study_id : int
        ID of the study associated with this cfmm2tar output.
    attached_tar_file : str, optional
        Name of the attached tar file.

    Raises
    ------
    Cfmm2tarError
        If cfmm2tar fails.
    """
    date_match = re.fullmatch(
        r"[a-zA-Z]+_[\w\-]+_(\d{8})_[\w\-]+_[\.a-zA-Z\d]+\.tar",
        tar_file,
    )
    if not date_match:
        msg = f"Output {tar_file} could not be parsed."
        raise Cfmm2tarError(msg)

    date_match = date_match.group(1)
    cfmm2tar = Cfmm2tarOutput(
        study_id=study_id,
        tar_file=tar_file,
        uid=uid.strip(),
        date=datetime(
            int(date_match[0:4]),
            int(date_match[4:6]),
            int(date_match[6:8]),
            tzinfo=TIME_ZONE,
        ),
        attached_tar_file=attached_tar_file,
    )
    db.session.add(cfmm2tar)
    db.session.commit()


def process_uid_file(uid_path: PathLike[str] | str):
    """Read a UID file, delete it, and return the UID."""
    path = pathlib.Path(uid_path)
    with path.open(encoding="utf-8") as uid_file:
        uid = uid_file.read()
    path.unlink()
    return uid


def find_studies_to_download(study, study_description, explicit_scans=None):
    """Find the studies to download, or override them with explicit scans."""
    existing_outputs = Cfmm2tarOutput.query.filter_by(study_id=study.id).all()
    if explicit_scans is not None:
        return [
            scan
            for scan in explicit_scans
            if scan["StudyInstanceUID"]
            not in {output.uid.strip() for output in existing_outputs}
        ]
    return [
        record
        for record in get_study_records(study, description=study_description)
        if record["StudyInstanceUID"]
        not in {output.uid.strip() for output in existing_outputs}
    ]


def check_tar_files(study_id, explicit_scans=None, user_id=None):
    """Launch cfmm2tar if there are any new tar files."""
    study = Study.query.get(study_id)
    user = User.query.get(user_id) if user_id is not None else None
    studies_to_download = find_studies_to_download(
        study,
        f"{study.principal}^{study.project_name}",
        explicit_scans,
    )
    if not studies_to_download:
        return
    new_studies = ", ".join(
        [new_study["PatientName"] for new_study in studies_to_download],
    )
    rq_job = Job.create(
        "autobidsportal.tasks.run_cfmm2tar",
        study_id,
        studies_to_download,
        timeout=app.config["CFMM2TAR_TIMEOUT"],
        connection=app.redis,
    )
    task = Task(
        id=rq_job.get_id(),
        name="run_cfmm2tar",
        description=f"Get tar files {new_studies} in study {study_id}",
        start_time=datetime.now(tz=TIME_ZONE),
        user=user,
        study_id=study_id,
    )
    db.session.add(task)
    db.session.commit()
    app.task_queue.enqueue_job(rq_job)


def handle_cfmm2tar(download_dir, study, target, dataset):
    """Run cfmm2tar on one target."""
    _, log = run_cfmm2tar_with_retries(
        str(download_dir),
        target["StudyInstanceUID"],
    )

    app.logger.info(
        "Successfully ran cfmm2tar for target %s.",
        target["PatientName"],
    )
    app.logger.info("Log: %s", log)

    if not (created_files := list(pathlib.Path(download_dir).iterdir())):
        msg = (
            f"No cfmm2tar results parsed for target {target}. "
            "Check the stderr for more information."
        )
        raise Cfmm2tarError(
            msg,
        )

    tar, uid_file, attached_tar = None, None, None
    for file_ in created_files:
        if file_.name.endswith(".attached.tar"):
            attached_tar = file_.name
        elif file_.name.endswith(".uid"):
            uid_file = file_
        elif file_.name.endswith(".tar"):
            tar = file_.name
        else:
            app.logger.warning("Unknown cfmm2tar output: %s", file_)
    if not tar:
        msg = "No tar file produced."
        raise Cfmm2tarError(msg)
    if not uid_file:
        msg = "No uid file produced."
        raise Cfmm2tarError(msg)

    created_files = list(set(created_files) - {uid_file})
    uid = process_uid_file(uid_file)

    with RiaDataset(
        download_dir,
        dataset.ria_alias,
        ria_url=dataset.custom_ria_url,
    ) as path_dataset:
        for file_ in created_files:
            copy2(file_, path_dataset / file_.name)
        finalize_dataset_changes(str(path_dataset), "Add new tar file.")
    record_cfmm2tar(
        tar,
        uid,
        study.id,
        attached_tar_file=attached_tar,
    )


def report_cfmm2tar(
    study_id: int,
    patient_names: Iterable[str],
    error_msgs: Iterable[str],
):
    send_email(
        "New cfmm2tar run",
        "\n".join(
            [
                (
                    "Attempted to download the following tar files "
                    f"for study {study_id}:"
                ),
            ]
            + [
                f"PatientName: {patient_name}"
                for patient_name in patient_names
            ]
            + ["\nErrors:\n"]
            + list(error_msgs),
        ),
        additional_recipients=[
            admin.email for admin in User.query.filter_by(admin=True).all()
        ]
        if error_msgs
        else None,
    )


def run_cfmm2tar(
    study: Study,
    studies_to_download: list[dict[str, str]],
    log_record: Loggable | None = None,
):
    """Run cfmm2tar for a given study.

    This will check which patients have already been downloaded, download any
    new ones, and record them in the database.

    Parameters
    ----------
    study_id : int
        ID of the study for which to run cfmm2tar.
    studies_to_download : list of dict, optional
        List of scans to get with cfmm2tar, where each scan is represented by
        a dict with keys "StudyInstanceUID" and "PatientName".
    """
    if not studies_to_download:
        return
    if not log_record:
        log_record = NullLogger()
    app.logger.info(
        "Running cfmm2tar for patients %s in study %i",
        [record["PatientName"] for record in studies_to_download],
        study.id,
    )

    dataset = ensure_dataset_exists(study.id, DatasetType.SOURCE_DATA)
    for target in studies_to_download:
        with tempfile.TemporaryDirectory(
            dir=app.config["CFMM2TAR_DOWNLOAD_DIR"],
        ) as download_dir:
            try:
                handle_cfmm2tar(download_dir, study, target, dataset)
            except Cfmm2tarError as err:
                app.logger.exception("cfmm2tar failed")
                log_record.log(str(err))
                continue


def find_unprocessed_tar_files(study_id):
    """Check for tar files that aren't in the dataset and add them."""
    study = Study.query.get(study_id)
    dataset = DataladDataset.query.filter_by(
        study_id=study.id,
        dataset_type=DatasetType.RAW_DATA,
    ).one_or_none()
    existing_tar_file_ids = (
        set()
        if dataset is None
        else {out.id for out in dataset.cfmm2tar_outputs}
    )
    new_tar_file_ids = {
        tar_file.id for tar_file in study.cfmm2tar_outputs
    } - existing_tar_file_ids
    if not new_tar_file_ids:
        return
    Task.launch_task(
        "run_tar2bids",
        "tar2bids run for all new tar files",
        study_id,
        list(new_tar_file_ids),
        study_id=study_id,
        timeout=app.config["TAR2BIDS_TIMEOUT"],
    )


def report_failed_tar2bids(tar_files: Iterable[str], err: str):
    send_email(
        "Failed tar2bids run",
        "\n".join(
            ["Tar2bids failed for tar files:", *list(tar_files)]
            + [
                (
                    "Note: Some of the tar2bids runs may have "
                    "completed. This email is sent if any of "
                    "them fail."
                ),
                "Error:",
                str(err),
            ],
        ),
        additional_recipients=[
            admin.email for admin in User.query.filter_by(admin=True).all()
        ],
    )


def report_successful_tar2bids(tar_files: Iterable[str], study: Study):
    send_email(
        "Successful tar2bids run.",
        "\n".join(
            ["Tar2bids successfully run for tar files:", *tar_files],
        ),
        additional_recipients={study.submitter_email}
        | {user.email for user in study.users_authorized},
    )


def run_tar2bids(
    study_id: int,
    tar_file_ids: list[int],
    logger: Loggable | None = None,
):
    """Run tar2bids for a specific study.

    Parameters
    ----------
    study_id : int
        ID of the study the tar files are associated with.

    tar_file_ids : list of int
        IDs of the tar files to be included in the tar2bids run.

    Raises
    ------
    Tar2bidsError
        If tar2bids fails.
    """
    if not tar_file_ids:
        return
    if not logger:
        logger = NullLogger()
    study = Study.query.get(study_id)
    cfmm2tar_outputs = [
        Cfmm2tarOutput.query.get(tar_file_id) for tar_file_id in tar_file_ids
    ]
    dataset_tar = ensure_dataset_exists(study_id, DatasetType.SOURCE_DATA)
    dataset_bids = ensure_dataset_exists(study_id, DatasetType.RAW_DATA)
    with tempfile.TemporaryDirectory(
        dir=app.config["TAR2BIDS_DOWNLOAD_DIR"],
    ) as bids_dir, tempfile.TemporaryDirectory(
        dir=app.config["TAR2BIDS_TEMP_DIR"],
    ) as temp_dir, tempfile.TemporaryDirectory(
        dir=app.config["CFMM2TAR_DOWNLOAD_DIR"],
    ) as download_dir, tempfile.NamedTemporaryFile(
        mode="w+",
        encoding="utf-8",
        buffering=1,
    ) as bidsignore:
        app.logger.info("Running tar2bids for study %i", study.id)
        if study.custom_bidsignore is not None:
            bidsignore.write(study.custom_bidsignore)
        for tar_out in cfmm2tar_outputs:
            with RiaDataset(
                download_dir,
                dataset_tar.ria_alias,
                ria_url=dataset_tar.custom_ria_url,
            ) as path_dataset_tar:
                tar_path = get_tar_file_from_dataset(
                    tar_out.tar_file,
                    path_dataset_tar,
                )
                try:
                    logger.log(
                        gen_utils().run_tar2bids(
                            Tar2bidsArgs(
                                output_dir=str(
                                    pathlib.Path(bids_dir) / "incoming",
                                ),
                                tar_files=[tar_path],
                                heuristic=study.heuristic,
                                patient_str=study.subj_expr,
                                temp_dir=temp_dir,
                                bidsignore=None
                                if study.custom_bidsignore is None
                                else bidsignore.name,
                                deface=study.deface,
                            ),
                        ),
                    )
                except Tar2bidsError as err:
                    app.logger.exception("tar2bids failed")
                    logger.log(str(err))
                    logger.log("Dataset contents:\n")
                    logger.log(
                        "\n".join(
                            render_dir_dict(
                                gen_dir_dict(
                                    str(pathlib.Path(bids_dir) / "incoming"),
                                    frozenset({".git", ".datalad"}),
                                ),
                            ),
                        ),
                    )

                    raise
            with RiaDataset(
                pathlib.Path(bids_dir) / "existing",
                dataset_bids.ria_alias,
                ria_url=dataset_bids.custom_ria_url,
            ) as path_dataset_study:
                merge_datasets(
                    pathlib.Path(bids_dir) / "incoming",
                    path_dataset_study,
                )
                finalize_dataset_changes(
                    path_dataset_study,
                    f"Ran tar2bids on tar file {tar_path}",
                )
                study.dataset_content = gen_dir_dict(
                    path_dataset_study,
                    frozenset({".git", ".datalad"}),
                )
                tar_out.datalad_dataset = dataset_bids
                db.session.commit()
        db.session.add(
            Tar2bidsOutput(
                study_id=study_id,
                cfmm2tar_outputs=cfmm2tar_outputs,
                bids_dir=None,
                heuristic=study.heuristic,
            ),
        )
        db.session.commit()


def archive_entire_dataset(
    path_dataset_raw,
    path_archive,
    dataset_id: int,
    repo: GitRepo,
):
    """Make a new archive of an entire dataset."""
    get_all_dataset_content(path_dataset_raw)
    archive_dataset(
        path_dataset_raw,
        path_archive,
    )
    return DatasetArchive(
        dataset_id=dataset_id,
        dataset_hexsha=repo.get_hexsha(),
        commit_datetime=datetime.fromtimestamp(
            repo.get_commit_date(date="committed"),
            tz=TIME_ZONE,
        ),
    )


def archive_partial_dataset(
    repo: GitRepo,
    latest_archive,
    path_archive,
    path_dataset_raw,
    dataset_id,
):
    """Make an archive of changed files since the latest archive."""
    updated_files = [
        path
        for path, entry in GitRepo(str(path_dataset_raw))
        .diff(latest_archive.dataset_hexsha, repo.get_hexsha())
        .items()
        if (entry["state"] in {"added", "modified"})
        and (entry["type"] in {"file", "symlink"})
    ]
    with ZipFile(path_archive, mode="x") as zip_file:
        for file_ in updated_files:
            get_tar_file_from_dataset(
                (archive_path := file_.relative_to(path_dataset_raw)),
                path_dataset_raw,
            )
            zip_file.write(file_, archive_path)
    return DatasetArchive(
        dataset_id=dataset_id,
        parent_id=latest_archive.id,
        dataset_hexsha=repo.get_hexsha(),
        commit_datetime=datetime.fromtimestamp(
            repo.get_commit_date(date="committed"),
            tz=TIME_ZONE,
        ),
    )


def archive_raw_data(study):
    """Clone a study dataset and archive it if necessary."""
    if (study.custom_ria_url is not None) or (study.dataset_content is None):
        return
    dataset_raw = ensure_dataset_exists(study.id, DatasetType.RAW_DATA)
    with tempfile.TemporaryDirectory(
        dir=app.config["TAR2BIDS_DOWNLOAD_DIR"],
    ) as dir_raw_data, RiaDataset(
        dir_raw_data,
        dataset_raw.ria_alias,
        ria_url=dataset_raw.custom_ria_url,
    ) as path_dataset_raw, tempfile.TemporaryDirectory(
        dir=app.config["TAR2BIDS_DOWNLOAD_DIR"],
    ) as dir_archive:
        latest_archive = max(
            dataset_raw.dataset_archives,
            default=None,
            key=lambda archive: archive.commit_datetime,
        )
        repo = GitRepo(str(path_dataset_raw))
        if (latest_archive) and (
            latest_archive.dataset_hexsha == repo.get_hexsha()
        ):
            app.logger.info("Archive for study %s up to date", study.id)
            return

        commit_datetime = datetime.fromtimestamp(
            repo.get_commit_date(date="committed"),
            tz=TIME_ZONE,
        )
        path_archive = pathlib.Path(dir_archive) / (
            f"{dataset_raw.ria_alias}_"
            f"{commit_datetime.isoformat().replace(':', '.')}_"
            f"{repo.get_hexsha()[:6]}.zip"
        )
        archive = (
            archive_entire_dataset(
                path_dataset_raw,
                path_archive,
                dataset_raw.id,
                repo,
            )
            if not latest_archive
            else archive_partial_dataset(
                repo,
                latest_archive,
                path_archive,
                path_dataset_raw,
                dataset_raw.id,
            )
        )
        make_remote_dir(
            app.config["ARCHIVE_BASE_URL"].split(":")[0],
            app.config["ARCHIVE_BASE_URL"].split(":")[1]
            + f"/{dataset_raw.ria_alias}",
        )
        copy_file(
            app.config["ARCHIVE_BASE_URL"],
            str(path_archive),
            f"/{dataset_raw.ria_alias}",
        )
    db.session.add(archive)
    db.session.commit()


def update_heuristics():
    """Clone the heuristic repo if it doesn't exist, then pull from it."""
    if subprocess.run(
        ["git", "-C", app.config["HEURISTIC_REPO_PATH"], "status"],
        check=False,
    ).returncode:
        app.logger.info("No heuristic repo present. Cloning it...")
        subprocess.run(
            [
                "git",
                "clone",
                app.config["HEURISTIC_GIT_URL"],
                app.config["HEURISTIC_REPO_PATH"],
            ],
            check=True,
        )

    app.logger.info("Pulling heuristic repo.")
    try:
        subprocess.run(
            ["git", "-C", app.config["HEURISTIC_REPO_PATH"], "pull"],
            check=True,
        )
    except subprocess.CalledProcessError:
        app.logger.exception("Pull from heuristic repo unsuccessful.")
        raise


def find_uncorrected_images(study_id):
    """Check for NIfTI images that haven't had gradcorrect applied."""
    # anat,func,fmap,dwi,asl
    study = Study.query.get(study_id)
    if study.scanner != "type2":
        return
    raw_dataset = DataladDataset.query.filter_by(
        study_id=study.id,
        dataset_type=DatasetType.RAW_DATA,
    ).one_or_none()
    if not raw_dataset:
        return
    derived_dataset = DataladDataset.query.filter_by(
        study_id=study.id,
        dataset_type=DatasetType.DERIVED_DATA,
    ).one_or_none()
    if not derived_dataset:
        Task.launch_task(
            "gradcorrect_study",
            "gradcorrect for new BIDS dataset",
            study_id,
            study_id=study_id,
            timeout=app.config["GRADCORRECT_TIMEOUT"],
        )
        return
    correctable_args = {
        "extension": "nii.gz",
        "datatype": ["anat", "func", "fmap", "dwi", "asl"],
    }
    with tempfile.TemporaryDirectory(
        dir=app.config["TAR2BIDS_DOWNLOAD_DIR"],
    ) as derivatives_dir, RiaDataset(
        derivatives_dir,
        derived_dataset.ria_alias,
        ria_url=derived_dataset.custom_ria_url,
    ) as path_dataset_derived:
        gradcorrect_path = path_dataset_derived / "gradcorrect"
        gradcorrect_path.mkdir(exist_ok=True)
        derived_layout = BIDSLayout(
            gradcorrect_path,
            validate=False,
        )
        corrected_files = {
            str(
                pathlib.Path(img.path).relative_to(
                    path_dataset_derived / "gradcorrect",
                ),
            )
            for img in derived_layout.get(**correctable_args)
        }
    with tempfile.TemporaryDirectory(
        dir=app.config["TAR2BIDS_DOWNLOAD_DIR"],
    ) as bids_dir, RiaDataset(
        bids_dir,
        raw_dataset.ria_alias,
        ria_url=raw_dataset.custom_ria_url,
    ) as path_dataset_raw:
        raw_layout = BIDSLayout(path_dataset_raw, validate=False)
        subjects = {
            img.get_entities()["subject"]
            for img in raw_layout.get(**correctable_args)
            if str(pathlib.Path(img.path).relative_to(path_dataset_raw))
            not in corrected_files
        }

    if not subjects:
        return
    Task.launch_task(
        "gradcorrect_study",
        f"gradcorrect run for subjects {subjects}",
        study_id,
        study_id=study_id,
        timeout=app.config["GRADCORRECT_TIMEOUT"],
        subject_labels=subjects,
    )


def run_gradcorrect(
    path_dataset_raw: PathLike[str] | str,
    path_out: PathLike[str] | str,
    subject_ids: Iterable[str] | None,
) -> None:
    """Run gradcorrect on a BIDS dataset, optionally on a subset of subjects."""
    participant_label = (
        ["--participant_label", *subject_ids] if subject_ids else []
    )
    apptainer_exec(
        [
            "/gradcorrect/run.sh",
            str(path_dataset_raw),
            str(path_out),
            "participant",
            "--grad_coeff_file",
            app.config["GRADCORRECT_COEFF_FILE"],
            *participant_label,
        ],
        app.config["GRADCORRECT_PATH"],
        app.config["GRADCORRECT_BINDS"].split(","),
    )


def gradcorrect_study(
    study_id: int,
    subject_labels: Iterable[str] | None = None,
) -> None:
    """Run gradcorrect on a set of subjects in a study."""
    dataset_bids = ensure_dataset_exists(study_id, DatasetType.RAW_DATA)
    dataset_derivatives = ensure_dataset_exists(
        study_id,
        DatasetType.DERIVED_DATA,
    )
    with tempfile.TemporaryDirectory(
        dir=app.config["TAR2BIDS_DOWNLOAD_DIR"],
    ) as bids_dir, tempfile.TemporaryDirectory(
        dir=app.config["TAR2BIDS_DOWNLOAD_DIR"],
    ) as derivatives_dir, RiaDataset(
        derivatives_dir,
        dataset_derivatives.ria_alias,
        ria_url=dataset_derivatives.custom_ria_url,
    ) as path_dataset_derivatives:
        with RiaDataset(
            bids_dir,
            dataset_bids.ria_alias,
            ria_url=dataset_bids.custom_ria_url,
        ) as path_dataset_bids:
            if subject_labels:
                for subject_label in subject_labels:
                    get_tar_file_from_dataset(
                        f"sub-{subject_label}",
                        path_dataset_bids,
                    )
            else:
                get_all_dataset_content(path_dataset_bids)
            run_gradcorrect(
                path_dataset_bids,
                path_dataset_derivatives / "gradcorrect",
                subject_labels,
            )
        rmtree(
            path_dataset_derivatives
            / "gradcorrect"
            / "sourcedata"
            / "scratch",
        )
        sub_string = (
            ",".join(subject_labels) if subject_labels else "all subjects"
        )
        finalize_dataset_changes(
            str(path_dataset_derivatives),
            f"Run gradcorrect on subjects {sub_string}",
        )


def archive_derivative_data(study: Study):
    """Clone a study dataset and archive it if necessary."""
    if study.custom_ria_url is not None:
        return
    dataset_derived = ensure_dataset_exists(study.id, DatasetType.DERIVED_DATA)
    with tempfile.TemporaryDirectory(
        dir=app.config["TAR2BIDS_DOWNLOAD_DIR"],
    ) as dir_derived_data, RiaDataset(
        dir_derived_data,
        dataset_derived.ria_alias,
        ria_url=dataset_derived.custom_ria_url,
    ) as path_dataset_derived, tempfile.TemporaryDirectory(
        dir=app.config["TAR2BIDS_DOWNLOAD_DIR"],
    ) as dir_archive:
        latest_archive = max(
            dataset_derived.dataset_archives,
            default=None,
            key=lambda archive: archive.commit_datetime,
        )
        repo = GitRepo(str(path_dataset_derived))
        if (latest_archive) and (
            latest_archive.dataset_hexsha == repo.get_hexsha()
        ):
            app.logger.info("Archive for study %s up to date", study.id)
            return

        commit_datetime = datetime.fromtimestamp(
            repo.get_commit_date(date="committed"),
            tz=TIME_ZONE,
        )
        path_archive = pathlib.Path(dir_archive) / (
            f"{dataset_derived.ria_alias}_"
            f"{commit_datetime.isoformat().replace(':', '.')}_"
            f"{repo.get_hexsha()[:6]}.zip"
        )
        archive = (
            archive_entire_dataset(
                path_dataset_derived,
                path_archive,
                dataset_derived.id,
                repo,
            )
            if not latest_archive
            else archive_partial_dataset(
                repo,
                latest_archive,
                path_archive,
                path_dataset_derived,
                dataset_derived.id,
            )
        )
        make_remote_dir(
            app.config["ARCHIVE_BASE_URL"].split(":")[0],
            app.config["ARCHIVE_BASE_URL"].split(":")[1]
            + f"/{dataset_derived.ria_alias}",
        )
        copy_file(
            app.config["ARCHIVE_BASE_URL"],
            str(path_archive),
            f"/{dataset_derived.ria_alias}",
        )
    db.session.add(archive)
    db.session.commit()
