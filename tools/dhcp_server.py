#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
极简 DHCP 服务器: 给直连的开发板分配 192.168.137.x 地址
(ICS 不可用时的替代方案; 只实现 DISCOVER->OFFER, REQUEST->ACK)

用法: python tools/dhcp_server.py
"""
import socket
import struct
import sys

SERVER_IP = '192.168.100.10'
BOARD_MAC = bytes.fromhex('0a4db38cd9f0')   # 鲁班猫3, 固定分 .130
BOARD_IP = '192.168.100.130'
POOL_START = 100
POOL_NET = '192.168.100'
MASK = '255.255.255.0'
LEASE = 86400

MAGIC = b'\x63\x82\x53\x63'


def ip2b(s):
    return socket.inet_aton(s)


def parse_options(data):
    opts = {}
    i = 0
    while i < len(data):
        t = data[i]
        if t == 255:
            break
        if t == 0:
            i += 1
            continue
        ln = data[i + 1]
        opts[t] = data[i + 2:i + 2 + ln]
        i += 2 + ln
    return opts


def build_reply(pkt, msg_type, yiaddr):
    xid = pkt[4:8]
    chaddr = pkt[28:44]
    rep = bytearray(236)
    rep[0] = 2            # BOOTREPLY
    rep[1] = 1            # ethernet
    rep[2] = 6            # hlen
    rep[4:8] = xid
    rep[10:12] = b'\x80\x00'   # broadcast flag
    rep[16:20] = ip2b(yiaddr)
    rep[20:24] = ip2b(SERVER_IP)
    rep[28:44] = chaddr
    rep += MAGIC
    rep += bytes([53, 1, msg_type])
    rep += bytes([54, 4]) + ip2b(SERVER_IP)
    rep += bytes([1, 4]) + ip2b(MASK)
    rep += bytes([3, 4]) + ip2b(SERVER_IP)
    rep += bytes([6, 4]) + ip2b(SERVER_IP)
    rep += bytes([51, 4]) + struct.pack('!I', LEASE)
    rep += bytes([255])
    return bytes(rep)


def main():
    pool_next = [POOL_START]
    leases = {}

    def alloc(mac):
        if mac == BOARD_MAC:
            return BOARD_IP
        if mac not in leases:
            leases[mac] = f'{POOL_NET}.{pool_next[0]}'
            pool_next[0] += 1
        return leases[mac]

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.bind((SERVER_IP, 67))
    print(f'DHCP server on {SERVER_IP}:67, board {BOARD_MAC.hex(":")} -> {BOARD_IP}', flush=True)

    while True:
        pkt, addr = s.recvfrom(2048)
        if len(pkt) < 240 or pkt[236:240] != MAGIC:
            continue
        opts = parse_options(pkt[240:])
        mtype = opts.get(53, b'')[0] if opts.get(53) else 0
        mac = pkt[28:34]
        offered = alloc(mac)
        if mtype == 1:      # DISCOVER
            print(f'DISCOVER from {mac.hex(":")} -> OFFER {offered}', flush=True)
            s.sendto(build_reply(pkt, 2, offered), ('255.255.255.255', 68))
        elif mtype == 3:    # REQUEST
            print(f'REQUEST  from {mac.hex(":")} -> ACK {offered}', flush=True)
            s.sendto(build_reply(pkt, 5, offered), ('255.255.255.255', 68))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
