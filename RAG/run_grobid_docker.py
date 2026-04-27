import argparse
import subprocess
import sys
import time
from pathlib import Path

import requests


DEFAULT_CONTAINER_NAME = "grobid"
DEFAULT_IMAGE = "grobid/grobid:0.8.1"
DEFAULT_HOST = "http://localhost"
DEFAULT_PORT = 8070


def run_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, capture_output=True, text=True)


def docker_is_available() -> bool:
    try:
        run_command(["docker", "--version"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def container_is_running(container_name: str) -> bool:
    result = run_command(
        [
            "docker",
            "ps",
            "--filter",
            f"name={container_name}",
            "--format",
            "{{.Names}}",
        ],
        check=False,
    )
    return any(line.strip() == container_name for line in result.stdout.splitlines())


def start_grobid_container(
    container_name: str,
    image: str,
    port: int,
) -> bool:
    if container_is_running(container_name):
        print(f"Container '{container_name}' is already running.")
        return False

    print(f"Starting GROBID container '{container_name}' from image '{image}'...")
    run_command(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            container_name,
            "-p",
            f"{port}:8070",
            image,
        ]
    )
    return True


def wait_for_grobid(base_url: str, timeout_seconds: int = 120) -> None:
    deadline = time.time() + timeout_seconds
    alive_url = f"{base_url}/api/isalive"

    while time.time() < deadline:
        try:
            response = requests.get(alive_url, timeout=5)
            if response.status_code == 200 and "true" in response.text.lower():
                print("GROBID is ready.")
                return
        except requests.RequestException:
            pass
        time.sleep(2)

    raise TimeoutError(f"GROBID did not become ready in {timeout_seconds} seconds.")


def stop_container(container_name: str) -> None:
    print(f"Stopping container '{container_name}'...")
    run_command(["docker", "stop", container_name], check=False)


def process_pdf(pdf_path: Path, output_dir: Path, base_url: str) -> None:
    output_file = output_dir / f"{pdf_path.stem}.grobid.tei.xml"
    endpoint = f"{base_url}/api/processFulltextDocument"

    with pdf_path.open("rb") as input_file:
        files = {
            "input": (pdf_path.name, input_file, "application/pdf"),
        }
        data = {
            "consolidateHeader": "1",
            "consolidateCitations": "0",
            "segmentSentences": "1",
        }
        response = requests.post(endpoint, files=files, data=data, timeout=300)

    if response.status_code != 200:
        raise RuntimeError(
            f"GROBID failed for '{pdf_path.name}' with HTTP {response.status_code}: {response.text[:300]}"
        )

    output_file.write_text(response.text, encoding="utf-8")
    print(f"Written: {output_file}")


def process_all_pdfs(input_dir: Path, output_dir: Path, base_url: str) -> None:
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF found in: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        process_pdf(pdf_file, output_dir, base_url)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run GROBID in Docker and convert PDF files to TEI XML."
    )
    parser.add_argument("--input-dir", type=Path, default=script_dir / "pdfs")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "results")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--container-name", default=DEFAULT_CONTAINER_NAME)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument(
        "--stop-container",
        action="store_true",
        help="Stop the GROBID container at the end if this script started it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not docker_is_available():
        print("Docker is not available. Install Docker Desktop and ensure it is running.")
        return 1

    base_url = f"{args.host}:{args.port}"
    started_by_script = False

    try:
        started_by_script = start_grobid_container(
            container_name=args.container_name,
            image=args.image,
            port=args.port,
        )

        wait_for_grobid(base_url)
        process_all_pdfs(args.input_dir, args.output_dir, base_url)

    except Exception as exc:  # broad exception for CLI error reporting
        print(f"Error: {exc}")
        return 1
    finally:
        if started_by_script and args.stop_container:
            stop_container(args.container_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
