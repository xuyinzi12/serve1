from __future__ import annotations

import html.parser
import subprocess
import urllib.parse
import urllib.request


class LinkParser(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value:
                self.links.append(value)


indexes = {
    "official": "https://pypi.org/simple/vllm/",
    "tuna": "https://pypi.tuna.tsinghua.edu.cn/simple/vllm/",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple/vllm/",
}

for name, index_url in indexes.items():
    with urllib.request.urlopen(index_url, timeout=20) as response:
        body = response.read().decode("utf-8")

    parser = LinkParser()
    parser.feed(body)
    candidates = [
        urllib.parse.urljoin(index_url, link)
        for link in parser.links
        if "vllm-0.26.0" in link and ".whl" in link
    ]
    if not candidates:
        print(f"{name} wheel=no")
        continue

    wheel_url = candidates[0].split("#", 1)[0]
    result = subprocess.run(
        [
            "curl",
            "--location",
            "--silent",
            "--show-error",
            "--max-time",
            "20",
            "--range",
            "0-16777215",
            "--output",
            "/dev/null",
            "--write-out",
            "%{http_code} %{time_total} %{size_download} %{speed_download}",
            wheel_url,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    metrics = result.stdout.strip()
    error = result.stderr.strip().replace("\n", " ")
    print(
        f"{name} curl_status={result.returncode} "
        f"http_time_bytes_bps={metrics} error={error}"
    )
