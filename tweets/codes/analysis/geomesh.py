#! /usr/bin/env python
# -*- encoding: utf-8 -*-


import math
from datetime import timedelta, datetime


def createnewfile(t):
    filename = t.strftime('%Y%m%d%H%M%S')
    return open(filename + ".csv", "wb")

def split_csvfile(filepath):
    """ 5分おきにCSVファイルを生成する """
    t = datetime.strptime("2015-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    f = createnewfile(t)

    for i in range(30 * 24 * 12):
        for line in csvfile:
            f.write(line)
            if t < line:
                t += timedelta(minutes=5)
                createnewfile(t)
                break


def get_first_mesh(lat, lng):
    left_operator = int(math.floor(lat * 15 / 10))
    right_operator = int(math.floor(lng - 100))
    sw_lat = left_operator / 15.0 * 10
    sw_lng = right_operator + 100

    return {
        'mesh_code': str(left_operator) + str(right_operator),
        'lat': sw_lat,
        'lng': sw_lng,
    }


def get_second_mesh(lat, lng):
    first_mesh_code = get_first_mesh(lat, lng)
    left_operator = int(math.floor(
                        (lat - first_mesh_code['lat']) * 100000 / 8333))
    right_operator = int(math.floor(
                         (lng - first_mesh_code['lng']) * 1000 / 125))
    sw_lat = left_operator * 8333. / 100000 + first_mesh_code['lat']
    sw_lng = right_operator * 125. / 1000 + first_mesh_code['lng']

    return {
        'mesh_code': (
            first_mesh_code['mesh_code'] +
            '-' +
            str(left_operator) +
            str(right_operator)),
        'lat': sw_lat,
        'lng': sw_lng,
    }


def get_third_mesh(lat, lng):
    second_mesh_code = get_second_mesh(lat, lng)
    left_operator = int(math.floor(
                        (lat - second_mesh_code['lat']) * 1000000 / 8333
                        ))
    right_operator = int(math.floor(
                         (lng - second_mesh_code['lng']) * 10000 / 125
                         ))
    sw_lat = left_operator * 8333. / 1000000 + second_mesh_code['lat']
    sw_lng = right_operator * 125. / 10000 + second_mesh_code['lng']

    return {
        'mesh_code': (
                      second_mesh_code['mesh_code'] +
                      '-' +
                      str(left_operator) +
                      str(right_operator)),
        'lat': sw_lat,
        'lng': sw_lng,
    }

# End of line
