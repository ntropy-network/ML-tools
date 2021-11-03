import subprocess
from os import listdir, rename, stat, path
from os.path import exists, isdir
import time
import requests


def notify(status, user, jobfile, return_code=None):
    SLACK_URL = "SLACK_CHANNEL_HOOK_URL" 
    TRANSLATION_TABLE = {
        ord("&"): "&amp;",
        ord("<"): "&lt;",
        ord(">"): "&gt;",
    }

    msg = f"JOB {'STARTING' if status == 'start' else 'DONE'} ({user}) {jobfile}"
    if return_code is not None:
        msg += f" [{return_code}]"

    print(msg, flush=True)

    msg = msg.translate(TRANSLATION_TABLE)
    requests.post(SLACK_URL, json={"text": msg})


MAX_RUNTIME = 3600 * 24
users = ["<LIST OF USERS>"]


def run_job(user, jobfile):
    process = subprocess.Popen(
        ["sudo", "-H", "-u", user, "/bin/bash", "-c", f"sh {jobfile}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    t0 = time.time()
    lines = []
    while True:
        stdout = process.stdout.readline().strip()
        lines.append(stdout)
        print(stdout, flush=True)
        if not exists(jobfile):
            process.terminate()
        return_code = process.poll()
        if return_code is not None:
            notify("stop", user, jobfile, return_code)
            with open(jobfile.replace(".sh", ".log"), "w") as fp:
                for line in lines:
                    fp.write(line + "\n")
            if exists(jobfile):
                tmp = jobfile.split("/")
                rename(jobfile, f"{'/'.join(tmp[:-1])}/_{tmp[-1]}")
            break
        else:
            if time.time() - t0 > MAX_RUNTIME:
                process.terminate()


while True:

    jobs = []
    for user in users:
        basedir = f"/home/{user}/jobs"
        if isdir(basedir):
            for job in listdir(basedir):
                if job[0] not in "_." and job[-3:] == ".sh":
                    jobfile = f"{basedir}/{job}"
                    jobs.append((user, jobfile, stat(jobfile).st_mtime))
    if jobs:
        user, jobfile, _ = min(jobs, key=lambda x: x[2])
        notify("start", user, jobfile)
        run_job(user, jobfile)
    else:
        print("QUEUE IS EMPTY", flush=True)

    time.sleep(3)
