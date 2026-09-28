#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Push local files to the LubanCat board via SFTP."""
import sys
import paramiko

HOST = '192.168.100.130'
USER = 'cat'
PASS = 'temppwd'


def main():
    if len(sys.argv) < 3:
        print('usage: board_push.py <local1> [local2 ...] <remote_dir>')
        sys.exit(1)
    *locals_, remote_dir = sys.argv[1:]
    t = paramiko.Transport((HOST, 22))
    t.connect(username=USER, password=PASS)
    sftp = paramiko.SFTPClient.from_transport(t)
    try:
        sftp.mkdir(remote_dir)
    except IOError:
        pass
    for local in locals_:
        import os
        remote = remote_dir.rstrip('/') + '/' + os.path.basename(local)
        print(f'{local} -> {remote}', flush=True)
        sftp.put(local, remote)
    sftp.close()
    t.close()
    print('done')


if __name__ == '__main__':
    main()
