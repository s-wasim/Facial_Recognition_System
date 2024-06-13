#import Face_Mesh
import re
from argparse import ArgumentParser
import numpy as np
import os
from Record_Face import capture_face, load_data
from Mark_Attendance import mark

def run_attendance_system():
    print('System is now Live')

def parse_args():
    parser = ArgumentParser(description='Module to run attendance System')
    parser.add_argument(
        '-r', '--rec_run', choices=['run', 'record', 'load'], type=str, required=True,
        help='Required command to dictate whether the server is in run or record mode'
    )
    parser.add_argument(
        '-n', '--name', type=str,
        help='Optional command to add new person with name (uses regex to verify name)'
    )
    parser.add_argument(
        '-l', '--loc', type=str,
        help='Location of images'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    try:
        if args.rec_run == 'run':
            assert args.name is None, 'Name should not be proviced with the "run" command'
            mark()
        elif args.rec_run == 'record':
            _name = args.name.strip()
            assert args.name != None, 'Name is required when recording'
            assert re.match(r'^[0-9]{4}-([A-Za-z]+ )*([A-Za-z]+)$', _name) is not None, 'Review name format'
            record_vector = capture_face(name=_name)
            os.mkdir(os.path.join('Dataset', _name))
            for key, val in record_vector.items():
                np.save(os.path.join('Dataset', _name, key), val)
            print(f'RECORED PERSON: {_name}')
        elif args.rec_run == 'load':
            assert args.loc != None, 'Location is required when loading'
            loc = args.loc
            assert os.path.exists(loc), 'Image folder does not exist'
            print('Loading Data...')
            load_data(loc)
            print('Read Data!')

    except AssertionError as e:
        print(f'ERROR: {e}')
