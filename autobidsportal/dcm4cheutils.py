"""Utilities for working with dcm4che and derived tools.

Define a (limited) Dcm4cheUtils class, which can query a DICOM server for
specified information. Adapted from YingLi Lu's class in the cfmm2tar
project.

For this to work, the machine must have dcm4che installed in some way (i.e.
natively or in a container).
"""
from __future__ import annotations

import logging
import pathlib
import pipes
import re
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from itertools import chain
from typing import Any, TypedDict

from defusedxml.ElementTree import parse
from flask import current_app

from autobidsportal.apptainer import ImageSpec, apptainer_exec


@dataclass
class DicomConnectionDetails:
    """Class for keeping track of details for connecting to a DICOM server."""

    connect: str
    use_tls: bool = True


@dataclass
class DicomCredentials:
    """Class for keeping track of a DICOM user's credentials."""

    username: str
    password: str


@dataclass
class DicomQueryAttributes:
    """Class containing attributes to be queried in a C-FIND request.

    At least one of the attributes must be assigned.

    Attributes
    ----------
    study_description : str, optional
        The StudyDescription to query. Passed to `findscu -m {}`.
    study_date : date, optional
        The date of the study to query. Converted to "YYYYMMDD" format and
        queries the "StudyDate" tag.
    patient_name : str, optional
        Search string for the patient names to retrieve.
    study_instance_uids : list of str
        Specific StudyInstanceUIDs to match.
    date_range_start : date, optional
        The start, inclusive, of a range of dates to query.
    date_range_end : date, optional
        The end, inclusive, of a range of dates to query.
    """

    study_description: str | None = None
    study_date: date | None = None
    patient_name: str | None = None
    study_instance_uids: Iterable[str] | None = None
    date_range_start: date | None = None
    date_range_end: date | None = None

    def __post_init__(self):
        """Check invariants of the args."""
        if all(
            [
                self.study_description is None,
                self.study_date is None,
                self.patient_name is None,
                self.study_instance_uids in [None, []],
                self.date_range_start is None,
                self.date_range_end is None,
            ],
        ):
            msg = (
                "You must specify at least one of study_description, study_date, or "
                "patient_name"
            )
            raise Dcm4cheError(
                msg,
            )
        if (self.study_date is not None) and (
            (self.date_range_start is not None)
            or (self.date_range_end is not None)
        ):
            msg = (
                "You may not define both study_date and either of date_range_start or "
                "date_range_end. Choose only one way to filter StudyDate."
            )
            raise Dcm4cheError(
                msg,
            )


@dataclass
class Cfmm2tarArgs:
    """A set of arguments for an invocation of cfmm2tar.

    Attributes
    ----------
    out_dir
        Directory to which to download tar files.
    study_instance_uid
        String specifying the StudyInstanceUid to download
    date_str
        String specifying the date(s) to download. Can include up to two
        dates and a "-" to indicate an open or closed interval of dates.
    patient_name
        PatientName string.
    project
        "Principal^Project" to search for.

    """

    out_dir: str
    study_instance_uid: str | None = None
    date_str: str | None = None
    patient_name: str | None = None
    project: str | None = None


@dataclass
class Tar2bidsArgs:
    """A set of arguments for an invocation of tar2bids.

    Attributes
    ----------
    tar_files
        Tar files on which to run tar2bids
    output_dir
        Directory for the output BIDS dataset
    """

    tar_files: Iterable[str]
    output_dir: str
    patient_str: str | None = None
    heuristic: str | None = None
    temp_dir: str | None = None
    bidsignore: str | None = None
    deface: bool = False


class DicomAttribute(TypedDict):
    """Dict representing one DICOM attribute."""

    tag_code: str
    tag_name: str
    tag_value: Any


def parse_findscu_xml(element_tree, output_fields) -> list[DicomAttribute]:
    """Find the relevant output from findscu output XML.

    Parameters
    ----------
    element_tree : ElementTree
        One XML document produced by findscu.
    output_fields : list of str
        List of output fields we're interested in (passed to findscu)

    Raises
    ------
    Dcm4cheError
        If dcm4che fails for any reason
    """
    output_fields = [
        f"{field[:8]}".upper()
        if re.fullmatch(r"[\dabcdefABCDEF]{8}", field)
        else field
        for field in output_fields
    ]
    out_list = []
    for field in output_fields:
        attribute_by_tag = element_tree.getroot().find(
            f"./DicomAttribute[@tag='{field}']",
        )
        attribute_by_keyword = element_tree.getroot().find(
            f"./DicomAttribute[@keyword='{field}']",
        )
        attribute = (
            attribute_by_tag
            if attribute_by_tag is not None
            else attribute_by_keyword
        )
        if attribute is None:
            msg = f"Missing expected output field {field} in findscu output"
            raise Dcm4cheError(
                msg,
            )
        tag_code = attribute.attrib["tag"]
        out_dict = {
            "tag_code": f"{tag_code[0:4]},{tag_code[4:8]}",
            "tag_name": attribute.attrib["keyword"],
        }
        if attribute.attrib["vr"] == "PN":
            value_elements = [
                element
                for element in attribute.findall(".//*")
                if element.text is not None
            ]
            if not value_elements:
                msg = f"Found PN attribute with no text: {attribute}"
                raise Dcm4cheError(
                    msg,
                )
            value = value_elements[0].text
        else:
            value = (
                value_attr.text
                if (value_attr := attribute.find("./Value")) is not None
                else ""
            )
        out_dict["tag_value"] = value
        out_list.append(out_dict)
    return out_list


class Dcm4cheUtils:
    """dcm4che utils."""

    def __init__(
        self,
        connection_details,
        credentials,
        cfmm2tar_spec,
        tar2bids_spec,
    ) -> None:
        """Set up the attrs for this utils instance."""
        self.logger = logging.getLogger(__name__)
        self.connect = connection_details.connect
        self.username = credentials.username
        self.password = credentials.password
        self.cfmm2tar_spec = cfmm2tar_spec
        self.tar2bids_spec = tar2bids_spec

        self._findscu_list = [
            "findscu",
            "--bind",
            "DEFAULT",
            "--connect",
            f"{self.connect}",
            "--accept-timeout",
            "10000",
            "--user",
            f"{pipes.quote(self.username)}",
            "--user-pass",
            f"{pipes.quote(self.password)}",
        ]

        if connection_details.use_tls:
            self._findscu_list.append("--tls-aes")

    def exec_cfmm2tar(self, cmd_list):
        """Execute the cfmm2tar container with the configured setup."""
        return apptainer_exec(
            cmd_list,
            self.cfmm2tar_spec.image_path,
            self.cfmm2tar_spec.binds,
            capture_output=True,
            text=True,
        )

    def get_all_pi_names(self):
        """Find all PIs the user has access to (by StudyDescription).

        Specifically, find all StudyDescriptions, take the portion before
        the caret, and return each unique value.
        """
        cmd = [*self._findscu_list, "-r", "StudyDescription"]

        try:
            completed_proc = self.exec_cfmm2tar(cmd)
        except subprocess.CalledProcessError as error:
            current_app.logger.error(
                "findscu failed while getting PI names: %s",
                error,
            )
            msg = "Non-zero exit status from findscu."
            raise Dcm4cheError(msg) from error
        err = completed_proc.stderr
        if err and err != "Picked up _JAVA_OPTIONS: -Xmx2048m\n":
            self.logger.error(err)

        dcm4che_out = completed_proc.stdout.splitlines()
        study_descriptions = [
            line for line in dcm4che_out if "StudyDescription" in line
        ]
        pi_matches = [
            re.match(r".*\[([\w ]+)\^[\w ]+\].*", line)
            for line in study_descriptions
        ]
        pis = [match.group(1) for match in pi_matches if match is not None]

        all_pis = list(
            set(pis) - set(current_app.config["DICOM_PI_BLACKLIST"]),
        )

        if len(all_pis) < 1:
            current_app.logger.error("findscu completed but no PIs found.")
            msg = "No PIs accessible."
            raise Dcm4cheError(msg)

        return all_pis

    def query_single_study(
        self,
        output_fields,
        attributes,
        retrieve_level="STUDY",
    ) -> list[list[DicomAttribute]]:
        """Query a DICOM server for specified tags from one study.

        Parameters
        ----------
        output_fields : list of str
            A list of DICOM tags to query (e.g. PatientName). Passed to
            `findscu -r {}`.
        attributes : DicomQueryAttributes
            A set of attributes to search for.
        retrieve_level : str
            Level at which to retrieve records. Defaults to "STUDY", but can
            also be "PATIENT", "SERIES", or "IMAGE".

        Returns
        -------
        list of list of dict
            A list containing one value for each result, where each result
            contains a list of dicts, where each dict contains the code, name,
            and value of each requested tag.

        Raises
        ------
        Dcm4cheError
            If dcm4che fails for any reason.
        """
        cmd = self._findscu_list.copy()

        if attributes.study_description is not None:
            cmd.extend(
                [
                    "-m",
                    f"StudyDescription={attributes.study_description}",
                ],
            )
        if attributes.study_date is not None:
            cmd.extend(
                [
                    "-m",
                    f'StudyDate={attributes.study_date.strftime("%Y%m%d")}',
                ],
            )
        elif (attributes.date_range_start is not None) or (
            attributes.date_range_end is not None
        ):
            start = (
                attributes.date_range_start.strftime("%Y%m%d")
                if attributes.date_range_start is not None
                else ""
            )
            end = (
                attributes.date_range_end.strftime("%Y%m%d")
                if attributes.date_range_end is not None
                else ""
            )
            cmd.extend(["-m", f"StudyDate={start}-{end}"])
        if attributes.patient_name is not None:
            cmd.extend(["-m", f"PatientName={attributes.patient_name}"])
        if attributes.study_instance_uids not in [None, []]:
            cmd.extend(
                [
                    "-m",
                    "StudyInstanceUID={}".format(
                        "\\\\".join(attributes.study_instance_uids),
                    ),
                ],
            )
        elif current_app.config["DICOM_SERVER_STUDYINSTANCEUID_WILDCARD"]:
            cmd.extend(["-m", "StudyInstanceUID=*"])

        cmd.extend(
            list(chain(*[["-r", f"{field}"] for field in output_fields])),
        )
        cmd.extend(["-L", f"{retrieve_level}"])

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd.extend(
                ["--out-dir", f"{tmpdir}", "--out-file", "000.xml", "-X"],
            )
            current_app.logger.info("Querying study with findscu.")
            try:
                completed_proc = self.exec_cfmm2tar(cmd)
            except subprocess.CalledProcessError as error:
                current_app.logger.error(
                    "Findscu failed while querying study.",
                )
                msg = "Non-zero exit status from findscu."
                raise Dcm4cheError(
                    msg,
                ) from error
            trees_xml = [
                parse(child) for child in pathlib.Path(tmpdir).iterdir()
            ]
        err = completed_proc.stderr
        if err and err != "Picked up _JAVA_OPTIONS: -Xmx2048m\n":
            self.logger.error(err)

        return [parse_findscu_xml(tree, output_fields) for tree in trees_xml]

    def run_cfmm2tar(
        self,
        args: Cfmm2tarArgs,
    ) -> tuple[list[list[str]], str]:
        """Run cfmm2tar with the given options.

        At least one of the optional search arguments must be provided.

        Parameters
        ----------
        args
            Arguments to pass to cfmm2tar

        Returns
        -------
            A list containing the tar file name and uid file name (in that
            order) for each result.

        Raises
        ------
        Cfmm2tarError
            If the arguments are malformed.
        Cfmm2tarTimeoutError
            If cfmm2tar times out.
        """
        if all(
            arg is None
            for arg in (
                args.study_instance_uid,
                args.date_str,
                args.patient_name,
                args.project,
            )
        ):
            msg = "At least one search argument must be provided."
            raise Cfmm2tarError(
                msg,
            )
        uid_query = [
            "-u",
            args.study_instance_uid
            if args.study_instance_uid is not None
            else [],
        ]
        project_query = (
            ["-p", args.project] if args.project is not None else []
        )

        with tempfile.NamedTemporaryFile(mode="w+", buffering=1) as cred_file:
            cred_file.write(self.username + "\n")
            cred_file.write(self.password + "\n")
            arg_list = (
                ["cfmm2tar"]
                + ["-c", cred_file.name]
                + uid_query
                + (["-d", args.date_str] if args.date_str is not None else [])
                + (
                    ["-n", args.patient_name]
                    if args.patient_name is not None
                    else []
                )
                + project_query
                + ["-s", current_app.config["DICOM_SERVER_URL"]]
                + [args.out_dir]
            )

            current_app.logger.info("Running cfmm2tar: %s", " ".join(arg_list))
            try:
                out = self.exec_cfmm2tar(arg_list)
            except subprocess.CalledProcessError as err:
                if "Timeout.java" in err.stderr:
                    current_app.logger.warning("cfmm2tar timed out.")
                    raise Cfmm2tarTimeoutError from err
                current_app.logger.error("cfmm2tar failed: %s", err.stderr)
                msg = f"Cfmm2tar failed:\n{err.stderr}"
                raise Cfmm2tarError(msg) from err

            all_out = out.stdout + out.stderr
            split_out = all_out.split("Retrieving #")[1:]

            current_app.logger.info("cfmm2tar stdout: %s", out.stdout)
            current_app.logger.info("cfmm2tar stderr: %s", out.stderr)

            tar_files = [
                [
                    line.split("created: ")[1]
                    for line in file_out.splitlines()
                    if any(
                        [
                            "tar file created" in line,
                            "uid file created" in line,
                        ],
                    )
                ]
                for file_out in split_out
            ]
            if tar_files == []:
                current_app.logger.warning("No tar files found for cfmm2tar.")
                if "Timeout.java" in all_out:
                    current_app.logger.warning("cfmm2tar timed out.")
                    raise Cfmm2tarTimeoutError

            return tar_files, all_out

    def run_tar2bids(
        self,
        args,
    ):
        """Run tar2bids with the given arguments.

        Returns
        -------
        The given output_dir, if successful.

        Raises
        ------
        Tar2bidsError
            If Tar2bids fails for any reason.
        """
        arg_list = (
            ["/opt/tar2bids/tar2bids"]
            + (
                ["-P", args.patient_str]
                if args.patient_str is not None
                else []
            )
            + (["-o", args.output_dir])
            + (["-h", args.heuristic] if args.heuristic is not None else [])
            + (["-w", args.temp_dir] if args.temp_dir is not None else [])
            + (["-b", args.bidsignore] if args.bidsignore is not None else [])
            + (["-D"] if args.deface else [])
            + list(args.tar_files)
        )

        current_app.logger.info("Running tar2bids.")
        try:
            out = apptainer_exec(
                arg_list,
                self.tar2bids_spec.image_path,
                self.tar2bids_spec.binds,
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                text=True,
            ).stdout
        except subprocess.CalledProcessError as err:
            current_app.logger.warning("tar2bids failed: %s", err.stdout)
            msg = f"Tar2bids failed:\n{err.stdout}"
            raise Tar2bidsError(msg) from err

        return out


def gen_utils():
    """Generate a Dcm4cheUtils with values from the current_app config."""
    return Dcm4cheUtils(
        DicomConnectionDetails(
            connect=current_app.config["DICOM_SERVER_URL"],
            use_tls=current_app.config["DICOM_SERVER_TLS"],
        ),
        DicomCredentials(
            current_app.config["DICOM_SERVER_USERNAME"],
            current_app.config["DICOM_SERVER_PASSWORD"],
        ),
        ImageSpec(
            current_app.config["CFMM2TAR_PATH"],
            current_app.config["CFMM2TAR_BINDS"].split(","),
        ),
        ImageSpec(
            current_app.config["TAR2BIDS_PATH"],
            current_app.config["TAR2BIDS_BINDS"].split(","),
        ),
    )


class Dcm4cheError(Exception):
    """Exception raised when something goes wrong with a dcm4che process."""


class Cfmm2tarError(Exception):
    """Exception raised when cfmm2tar fails."""


class Cfmm2tarTimeoutError(Exception):
    """Exception raised when cfmm2tar times out."""


class Tar2bidsError(Exception):
    """Exception raised when tar2bids fails."""
