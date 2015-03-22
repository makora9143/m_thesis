#! /usr/bin/env python
# -*- encoding: utf-8 -*-


import pandas as pd


def main(filepath):
    csv = pd.read_csv(filepath,
                      names=[
                            'id', 'username', 'account', 'userId',
                            'text', 'lat', 'lng', 'timestamp',
                            'meshcode'],
                      delimiter=',',
                      quotechar='"')

    print csv



if __name__ == '__main__':
    filepath = "../csvfiles/modify_data/tokyo_meshcode.csv"
    main(filepath)


# End of Line.
