#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:23:46 2024

@author: tiehanpan
"""


def read_text(filename):

    data = open(filename, 'r')
    list_str = data.readlines()
        
    return(list_str)

read_text('nyc_temp_2010.txt')

