#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import torch as t
import torch
import os


def gen_golden_data_simple():
    input_x = np.random.uniform(-3, 3, [128]).astype(np.float16)
    result=t.tensor(input_x)
    result = t.logsumexp(result,0)
    golden = (result.numpy()).astype(np.float16)
    # print(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")

    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()