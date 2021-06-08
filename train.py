#-*- coding: utf-8 -*-
import torch
import torch.utils.data as data_utils
import signal
import sys
import os
import logging
import math
import json
import time
import trimesh
import argparse
from mesh_to_sdf import sample_sdf_near_surface,get_surface_point_cloud
import Network
import utils
from torch.utils.data import Dataset,DataLoader
import time
import torch.optim as optim
from mesh import create_mesh

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', default='/mnt/data3/lj/transparent/Data/Shapes/train/Shape__0/visualHullSubd_20.ply', help='path to images' )
parser.add_argument('--batchsize', type=int,default=100000, help='the number of sampled SDFs for each training iteration')
parser.add_argument('--epoch', type=int,default=10000, help='epoch number')
parser.add_argument('--ckpt_path',default='./ckpt', help='checkpoint path')