#! /usr/bin/env python
# -*- encoding: utf-8 -*-


import pandas as pd
import csv
from geomesh import get_third_mesh


def main(filepath):
    csv = pd.read_csv(filepath,
                      names=[
                            'id', 'username', 'account', 'userId',
                            'text', 'lat', 'lng', 'timestamp',
                            'meshcode'],
                      delimiter=',',
                      quotechar='"')

    print csv


def append_grid(inputfile, outputfile):
    f = open(inputfile)
    reader = csv.reader(f, delimiter=',', quotechar='"')
    tweetWriter = csv.writer(open(outputfile, 'wb'),
                             delimiter=',',
                             quotechar='"',
                             quoting=csv.QUOTE_ALL)
    tweetWriter.writerow(["tweet_id", "screen_name", "account", "user_id", "text", "lat", "lng", "timestamp", "1st_mesh", "2nd_mesh", "3rd_mesh"])
    for row in reader:
        if len(row) == 9:
            tweetWriter.writerow(row + get_third_mesh(float(row[6]), float(row[7]))['mesh_code'].split('-'))
    return True


if __name__ == '__main__':
    inputfile = '../csvfiles/raw_data/tokyo-20150101-20150131.csv'
    outputpath = "../csvfiles/modify_data/tokyo_meshcode.csv"

    append_grid(inputfile, outputpath)


# End of Line.
