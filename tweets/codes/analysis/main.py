#! /usr/bin/env python
# -*- encoding: utf-8 -*-


import pandas as pd
import csv
from geomesh import get_third_mesh
from datetime import timedelta, datetime


def main(filepath):
    csv = pd.read_csv(filepath,
                      names=[
                            'id', 'username', 'account', 'userId',
                            'text', 'lat', 'lng', 'timestamp',
                            'meshcode'],
                      delimiter=',',
                      quotechar='"')

    print csv


def append_grid(inputfile):
    reader = csv.reader(open(inputfile), delimiter=',', quotechar='"')
    header = ["tweet_id", "screen_name", "account", "user_id", "text", "picture", "lat", "lng", "timestamp", "1st_mesh", "2nd_mesh", "3rd_mesh"]
    t = datetime.strptime("2015-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    f = open("csvfiles/" +t.strftime('%Y%m%d%H%M%S') + ".csv", "wb")
    tweetWriter = csv.writer(f,
                             delimiter=',',
                             quotechar='"',
                             quoting=csv.QUOTE_ALL)
    tweetWriter.writerow(header)
    for row in reader:
        if len(row) == 9:
            if datetime.strptime(row[8], "%Y-%m-%d %H:%M:%S") > t:
                t += timedelta(minutes=5)
                f = open("csvfiles/" + t.strftime('%Y%m%d%H%M') + ".csv", "wb")
                tweetWriter = csv.writer(f,
                                         delimiter=',',
                                         quotechar='"',
                                         quoting=csv.QUOTE_ALL)
                tweetWriter.writerow(header)
            tweetWriter.writerow(row + get_third_mesh(float(row[6]), float(row[7]))['mesh_code'].split('-'))
    return True


if __name__ == '__main__':
    inputfile = '../../csvfiles/raw_data/tokyo-20150101-20150131.csv'
    outputpath = "../../csvfiles/modify_data/tokyo_meshcode.csv"

    append_grid(inputfile)


# End of Line.
