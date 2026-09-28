#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SSH helper for LubanCat board: run commands / transfer files with password auth."""
import sys
import paramiko

HOST = '192.168.100.130'
USER = 'cat'
PASS = 'temppwd'


def get_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=15)
    return c


def run(cmd, timeout=300):
    c = get_client()
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode('utf-8', 'replace')
    err = stderr.read().decode('utf-8', 'replace')
    code = stdout.channel.recv_exit_status()
    c.close()
    return code, out, err


if __name__ == '__main__':
    code, out, err = run(' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'uname -a')
    sys.stdout.write(out)
    if err:
        sys.stderr.write(err)
    sys.exit(code)
