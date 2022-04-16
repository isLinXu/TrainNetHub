import numpy as np
import torch
import argparse
import os, time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--idle', action='store_true', help='set to idle state without running')

args = parser.parse_args()


def main():
    print('Allocate from GPU:', args.gpu)
    # ================================================

    gpuid = int(args.gpu)
    data = []

    device = torch.device('cuda:%d' % gpuid)
    total, used = os.popen(
        'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    ).read().split('\n')[gpuid].split(',')
    total = int(total)
    used = int(used)

    print('GPU:%d mem:' % gpuid, total, 'used:', used)

    try:
        block_mem = (total - used) // 4
        # print(block_mem)
        x = torch.rand((int(block_mem), 256, 1024)).to(device)
    except RuntimeError as err:
        del x
        print(err)
        block_mem = (total - used) // 5
        # print(block_mem)
        x = torch.rand((int(block_mem), 256, 1024)).to(device)

    data.append(x)

    result = os.popen(
        'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
    ).read().split('\n')
    print('gpu mem before running:', result)
    # ================================================

    while True:

        # if not idle, run some boring op to consume time and gpu resource
        if not args.idle:
            x.mul_(0.1).mul_(10)

        time.sleep(3)

        result = os.popen(
            'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
        ).read().split('\n')
        print('real-time running gpu mem:', result, end='\r')

        total, used = os.popen(
            'nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
        ).read().split('\n')[gpuid].split(',')
        total = int(total)
        used = int(used)

        counter = (total - used) // 1024
        if counter >= 1:
            # print(block_mem)
            tmp = torch.randn(counter, 256, 1024, 1024).to(device)  # 1GB
            data.append(tmp)
            print('\nseized %d GB memory...^-^...' % counter)


if __name__ == '__main__':
    main()