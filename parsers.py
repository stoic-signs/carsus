import numpy as np
import pandas as pd
import itertools
import warnings
import os
import sys


class BaseParser(object):
    def __init__(self, input_data=None):
        self.base = None
        if input_data i not None:
            self.load(input_data)

    @abstractmethod
    def load(self, input_data):
        pass

        def __call__(self, input_data):
            self.load(input_data)


self.def find_row(file, string, num_row=False):

    with open(file) as File:
        n = 0
        for line in File:
            n += 1
            if string in line:
                break
    if num_row is True:
        return (n - 1)
    return line


def search_header(file, keys, start=0, stop=50):
    head = {k.strip('!'): None for k in keys}
    with open(file, encoding='ISO-8859-1') as File:
        for line in itertools.islice(f, start, stop):
            for k in keys:
                if k.lower() in line.lower():
                    head[k.strip('!')] = line.split()[0]
    return head


def to_float(string):

    try:
        value = float(string.replace('D', 'E'))

    except ValueError:

        if string == '1-.00':      # Bad value at MG/VIII/23oct02/phot_sm_3000 line 23340
            value = 10.00

        if string == '*********':  # Bad values at SUL/V/08jul99/phot_op.big lines 9255-9257
            value = np.nan

    return value


class CMFGENEnergyLevels(BaseParser):
    keys = ["!Date", '!Format date', 'Number of energy levels',
            'Ionization energy', '!Screened nuclear charge', 'Number of transitions']

    def load(self, fname):
        head = search_header(fname, self.keys)
        kwargs = {}
        kwargs['header'] = None
        kwargs['index_col'] = False
        kwargs['sep'] = '\s+'
        kwargs['skiprows'] = find_row(
            fname, "Number of transitions", num_row=True)
        n = int(head['Number or energy levels'])
        kwargs['nrows'] = n
        columns = ['Config', 'g', 'E(cm^-1)', 'ev', 'Hz 10^15', 'Lam(A)']

        try:
            df = pd.read_csv(fname, **kwargs, engine='python')

        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
            warnings.warn('Empty Table')
        if df.shape[1] == 10:
            # Read column names and split them keeping one space (e.g. '10^15 Hz')
            columns = find_row(fname, 'E(cm^-1)', "Lam").split('  ')
            # Filter list elements containing empty strings
            columns = [c for c in columns if c != '']
            # Remove left spaces and newlines
            columns = [c.rstrip().lstrip() for c in columns]
            columns = ['Configuration'] + columns
            df.columns = columns

        elif df.shape[1] == 7:
            df.columns = columns + ['#']
            df = df.drop(columns=['#'])

        elif df.shape[1] == 6:
            df.columns = ['Configuration', 'g',
                          'E(cm^-1)', 'Hz 10^15', 'Lam(A)', '#']
            df = df.drop(columns=['#'])

        elif df.shape[1] == 5:
            df.columns = columns[:-2] + ['#']
            df = df.drop(columns=['#'])

        else:
            # TODO: raise exception here (discuss)
            warnings.warn('Inconsistent number of columns')

        self.fname = fname
        self.base = df
        self.columns = df.columns.tolist()
        self.head = head


class CMFGENOscStrengths(BaseParser):
    keys = CMFGENEnergyLevels.keys

    def load(self, fname):
        meta = parse_header(fname, self.keys)
        kwargs = {}
        kwargs['header'] = None
        kwargs['index_col'] = False
        kwargs['sep'] = '\s*\|\s*|-?\s+-?\s*|(?<=[^ED\s])-(?=[^\s])'
        # kwargs['sep']='(?<=[^E])-(?:[ ]{1,})?|(?<!-)[ ]{2,}[-,\|]?'
        kwargs['skiprows'] = find_row(
            fname, "Transition", "Lam", num_row=True) + 1

        n = int(meta['Number of transitions'])
        kwargs['nrows'] = n

        columns = ['State A', 'State B', 'f', 'A',
                   'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']

        try:
            df = pd.read_csv(fname, **kwargs, engine='python')

        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
            warnings.warn('Empty table')

        if df.shape[1] == 9:
            df.columns = columns

        elif df.shape[1] == 10:
            df.columns = columns + ['?']
            df = df.drop(columns=['?'])

        elif df.shape[1] == 8:
            df.columns = columns[:-2] + ['#']
            df = df.drop(columns=['#'])
            df['Lam(obs)'] = np.nan
            df['% Acc'] = np.nan

        else:
            warnings.warn('Inconsistent number of columns')

        if df.shape[0] > 0 and 'D' in str(df['f'][0]):
            df['f'] = df['f'].map(to_float)
            df['A'] = df['A'].map(to_float)

        self.fname = fname
        self.base = df
        self.columns = df.columns.tolist()
        self.head = head
