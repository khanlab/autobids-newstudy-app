#!/usr/bin/env python
"""
Define a (limited) Dcm4cheUtils class, which can query a DICOM server for
specified information. Adapted from YingLi Lu's class in the cfmm2tar
project.
For this to work, the machine must have dcm4che installed in some way (i.e.
natively or in a container).
"""

import subprocess
import logging
import re
import tempfile
import pathlib
from dataclasses import dataclass
from datetime import date
from typing import Sequence

# for quote python strings for safe use in posix shells
import pipes
from defusedxml.ElementTree import parse

from flask import current_app


def _get_stdout_stderr_returncode(cmd):
    """
    Execute the external command and get its stdout, stderr and return code
    """
    proc = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        shell=True,  # This is kind of unsafe in a webapp, should rethink
    )

    return proc.stdout, proc.stderr, proc.returncode


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

    study_description: str = None
    study_date: date = None
    patient_name: str = None
    study_instance_uids: Sequence[str] = None
    date_range_start: date = None
    date_range_end: date = None

    def __post_init__(self):
        if all(
            [
                self.study_description is None,
                self.study_date is None,
                self.patient_name is None,
                self.study_instance_uids in [None, []],
                self.date_range_start is None,
                self.date_range_end is None,
            ]
        ):
            raise Dcm4cheError(
                "You must specify at least one of study_description, "
                "study_date, or patient_name"
            )
        if (self.study_date is not None) and (
            (self.date_range_start is not None)
            or (self.date_range_end is not None)
        ):
            raise Dcm4cheError(
                "You may not define both study_date and either of "
                "date_range_start or date_range_end. Choose only one way to "
                "filter StudyDate."
            )


@dataclass
class Tar2bidsArgs:
    """A set of arguments for an invocation of tar2bids.

    Attributes
    ----------
    tar_files : list of str
        Tar files on which to run tar2bids
    output_dir : str
        Directory for the output BIDS dataset
    """

    tar_files: Sequence[str]
    output_dir: str
    patient_str: str = None
    heuristic: str = None
    temp_dir: str = None


def parse_findscu_xml(element_tree, output_fields):
    """Find the relevant output from findscu output XML.

    Parameters
    ----------
    element_tree : ElementTree
        One XML document produced by findscu.
    output_fields : list of str
        List of output fields we're interested in (passed to findscu)
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
            f"./DicomAttribute[@tag='{field}']"
        )
        attribute_by_keyword = element_tree.getroot().find(
            f"./DicomAttribute[@keyword='{field}']"
        )
        attribute = (
            attribute_by_tag
            if attribute_by_tag is not None
            else attribute_by_keyword
        )
        if attribute is None:
            raise Dcm4cheError(
                f"Missing expected output field {field} in findscu output"
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
            if len(value_elements) == 0:
                raise Dcm4cheError(
                    f"Found PN attribute with no text: {attribute}"
                )
            value = value_elements[0].text
        else:
            value = attribute.find("./Value").text
        out_dict["tag_value"] = value
        out_list.append(out_dict)
    return out_list


class Dcm4cheUtils:
    """
    dcm4che utils
    """

    def __init__(
        self,
        connection_details,
        credentials,
        dcm4che_path="",
        tar2bids_path="",
    ):
        self.logger = logging.getLogger(__name__)
        self.connect = connection_details.connect
        self.username = credentials.username
        self.password = credentials.password
        self.dcm4che_path = dcm4che_path

        self._findscu_str = (
            f"{self.dcm4che_path} findscu"
            + " --bind  DEFAULT"
            + f" --connect {self.connect}"
            + " --accept-timeout 10000 "
            + f" --user {pipes.quote(self.username)} "
            + f" --user-pass {pipes.quote(self.password)} "
        )
        if connection_details.use_tls:
            self._findscu_str += " --tls-aes "
        self._tar2bids_list = f"{tar2bids_path}tar2bids".split()

    def get_all_pi_names(self):
        """Find all PIs the user has access to (by StudyDescription).
        Specifically, find all StudyDescriptions, take the portion before
        the caret, and return each unique value."""
        cmd = self._findscu_str + " -r StudyDescription "

        try:
            out, err, _ = _get_stdout_stderr_returncode(cmd)
        except subprocess.CalledProcessError as error:
            current_app.logger.error(
                "findscu failed while getting PI names: %s", error
            )
            raise Dcm4cheError("Non-zero exit status from findscu.") from error
        if err and err != "Picked up _JAVA_OPTIONS: -Xmx2048m\n":
            self.logger.error(err)

        dcm4che_out = str(out, encoding="utf-8").splitlines()
        study_descriptions = [
            line for line in dcm4che_out if "StudyDescription" in line
        ]
        pi_matches = [
            re.match(r".*\[([\w ]+)\^[\w ]+\].*", line)
            for line in study_descriptions
        ]
        pis = [match.group(1) for match in pi_matches if match is not None]

        all_pis = list(
            set(pis) - set(current_app.config["DICOM_PI_BLACKLIST"])
        )

        if len(all_pis) < 1:
            current_app.log.error("findscu completed but no PIs found.")
            raise Dcm4cheError("No PIs accessible.")

        return all_pis

    def query_single_study(
        self,
        output_fields,
        attributes,
        retrieve_level="STUDY",
    ):
        """Queries a DICOM server for specified tags from one study.
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
        """
        cmd = self._findscu_str

        if attributes.study_description is not None:
            cmd = (
                f"{cmd} -m "
                f'StudyDescription="{attributes.study_description}"'
            )
        if attributes.study_date is not None:
            cmd = (
                f"{cmd} "
                f'-m StudyDate="{attributes.study_date.strftime("%Y%m%d")}"'
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
            cmd = f'{cmd} -m StudyDate="{start}-{end}"'
        if attributes.patient_name is not None:
            cmd = f'{cmd} -m PatientName="{attributes.patient_name}"'
        if attributes.study_instance_uids not in [None, []]:
            cmd = '{} -m StudyInstanceUID="{}"'.format(
                cmd, "\\\\".join(attributes.study_instance_uids)
            )
        else:
            cmd = f'{cmd} -m StudyInstanceUID="*"'

        cmd = " ".join([cmd] + [f"-r {field}" for field in output_fields])
        cmd = f"{cmd} -L {retrieve_level}"

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = f"{cmd} --out-dir {tmpdir} --out-file 000.xml -X"
                current_app.logger.info("Querying study with findscu.")
                _, err, _ = _get_stdout_stderr_returncode(cmd)

                trees_xml = [
                    parse(child) for child in pathlib.Path(tmpdir).iterdir()
                ]
        except subprocess.CalledProcessError as error:
            current_app.logger.error("Findscu failed while querying study.")
            raise Dcm4cheError("Non-zero exit status from findscu.") from error

        if err and err != "Picked up _JAVA_OPTIONS: -Xmx2048m\n":
            self.logger.error(err)

        return [parse_findscu_xml(tree, output_fields) for tree in trees_xml]

    def run_cfmm2tar(
        self, out_dir, date_str=None, patient_name=None, project=None
    ):
        """Run cfmm2tar with the given options.
        At least one of the optional search arguments must be provided.
        Arguments
        ---------
        out_dir : str
            Directory to which to download tar files.
        date_str : str, optional
            String specifying the date(s) to download. Can include up to two
            dates and a "-" to indicate an open or closed interval of dates.
        patient_name : str, optional
            PatientName string.
        project : str, optional
            "Principal^Project" to search for.

        Returns
        -------
        list of list of str
            A list containing the tar file name and uid file name (in that
            order) for each result.
        """
        if all(arg is None for arg in [date_str, patient_name, project]):
            raise Cfmm2tarError(
                "At least one search argument must be provided."
            )
        date_query = ["-d", date_str] if date_str is not None else []
        name_query = ["-n", patient_name] if patient_name is not None else []
        project_query = ["-p", project] if project is not None else []

        with tempfile.NamedTemporaryFile(mode="w+", buffering=1) as cred_file:
            cred_file.write(self.username + "\n")
            cred_file.write(self.password + "\n")
            arg_list = (
                self.dcm4che_path.split()
                + ["cfmm2tar"]
                + ["-c", cred_file.name]
                + date_query
                + name_query
                + project_query
                + [out_dir]
            )

            try:
                current_app.logger.info(
                    "Running cfmm2tar: %s", " ".join(arg_list)
                )
                out = subprocess.run(
                    arg_list,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as err:
                if "Timeout.java" in err.stderr:
                    current_app.logger.warning("cfmm2tar timed out.")
                    raise Cfmm2tarTimeoutError() from err
                current_app.logger.error("cfmm2tar failed: %s", err.stderr)
                raise Cfmm2tarError(f"Cfmm2tar failed:\n{err.stderr}") from err

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
                        ]
                    )
                ]
                for file_out in split_out
            ]
            if tar_files == []:
                current_app.logger.warning("No tar files found for cfmm2tar.")
                if "Timeout.java" in all_out:
                    current_app.logger.warning("cfmm2tar timed out.")
                    raise Cfmm2tarTimeoutError()

            return tar_files

    def run_tar2bids(
        self,
        args,
    ):
        """Run tar2bids with the given arguments.
        Returns
        -------
        The given output_dir, if successful.
        """
        arg_list = (
            self._tar2bids_list
            + (
                ["-P", args.patient_str]
                if args.patient_str is not None
                else []
            )
            + (["-o", args.output_dir])
            + (["-h", args.heuristic] if args.heuristic is not None else [])
            + (["-w", args.temp_dir] if args.temp_dir is not None else [])
            + args.tar_files
        )
        try:
            current_app.logger.info("Running tar2bids.")
            subprocess.run(arg_list, check=True)
        except subprocess.CalledProcessError as err:
            current_app.logger.warning("tar2bids failed: %s", err.stderr)
            raise Tar2bidsError(f"Tar2bids failed:\n{err.stderr}") from err

        return args.output_dir


def gen_utils(tar2bids_img=None):
    """Generate a Dcm4cheUtils with values from the current_app config."""
    tar2bids_path = (
        str(
            pathlib.Path(current_app.config["TAR2BIDS_IMAGE_DIR"])
            / tar2bids_img
        )
        if tar2bids_img is not None
        else ""
    )
    return Dcm4cheUtils(
        DicomConnectionDetails(
            connect=current_app.config["DICOM_SERVER_URL"],
            use_tls=current_app.config["DICOM_SERVER_TLS"],
        ),
        DicomCredentials(
            current_app.config["DICOM_SERVER_USERNAME"],
            current_app.config["DICOM_SERVER_PASSWORD"],
        ),
        current_app.config["DCM4CHE_PREFIX"],
        " ".join([current_app.config["TAR2BIDS_PREFIX"], tar2bids_path]),
    )


class Dcm4cheError(Exception):
    """Exception raised when something goes wrong with a dcm4che process."""

    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message


class Cfmm2tarError(Exception):
    """Exception raised when cfmm2tar fails."""

    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message


class Cfmm2tarTimeoutError(Exception):
    """Exception raised when cfmm2tar times out."""


class Tar2bidsError(Exception):
    """Exception raised when tar2bids fails."""

    def __init__(self, message):
        super().__init__()
        self.message = message

    def __str__(self):
        return self.message
