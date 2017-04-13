"""A set of helper functions to work with the astropy module."""
import functools
import random
import string
import tempfile
import subprocess
import collections
from itertools import cycle, islice

import numpy as np
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.vizier import Vizier

###############################################################################
# Astropy Utilities                                                           #
###############################################################################

def change_column_dtype(table, colname, newdtype):
    '''Changes the dtype of a column in a table.

    Use this function to change the dtype of a particular column in a table.
    '''
    tempcol = table[colname]
    colindex = table.colnames.index(colname)
    del(table[colname])
    table.add_column(np.asanyarray(tempcol, dtype=newdtype), index=colindex)

def astropy_table_index(table, column, value):
    '''Returns the row index of the table which has the value in column.

    There are often times when you want to know the index of the row
    where a certain column has a value. This function will return a 
    list of row indices that match the value in the column.'''
    return astropy_table_indices(table, column, [value])

def astropy_table_indices(table, column, values):
    '''Returns the row indices of the table which have the values in column.

    If you need to get the indices of values located in the column of a table,
    this function will determine that for you.
    '''
    indices = mark_selections_in_columns(table[column], values)
    return np.where(indices)

def mark_selections_in_columns(col, values):
    '''Return index indicating values are in col.

    Returns an index array which is the size of col that indicates True when
    col holds an entry equal to value, and False otherwise.'''
    if len(col) > len(values)**2:
        return multi_logical_or(*[col == v for v in values])
    else:
        try:
            valset = set(values)
        except TypeError:
            unmasked_values = values[values.mask == False]
            valset = set(unmasked_values)
        index = []
        for v in col:
            try:
                incol = v in valset
            except TypeError:
                incol = False
            index.append(incol)
        return index

def multi_logical_or(*arrs):
    '''Performs a logical or for an arbitrary number of boolean arrays.'''
    return functools.reduce(np.logical_or, arrs, False)

def multi_logical_and(*arrs):
    '''Performs a logical or for an arbitrary number of boolean arrays.'''
    return functools.reduce(np.logical_and, arrs, True)

def astropy_table_row(table, column, value):
    '''Returns the row of the table which has the value in column.

    If you want to know the row in an astropy table where a value in a
    column corresponds to a given value, this function will return that
    row. If there are multiple rows which match the value in the 
    column, you will get all of them. If no rows match the value, this
    function will throw a ValueError.'''
    return table[astropy_table_index(table, column, value)]

def extract_subtable_from_column(table, column, selections):
    '''Returns a table which only contains values in selections.

    This function will create a Table whose values in column are only
    those found in selections.
    '''
    return table[astropy_table_indices(table, column, selections)]

def filter_column_from_subtable(table, column, selections):
    '''Returns a table where none of the values in column are selections.

    This function will create a Table whose values are those in column which
    are not found in selections.
    '''

    subindices = astropy_table_indices(table, column, selections)
    print(subindices)
    compindices = get_complement_indices(subindices, len(table))
    print(compindices)
    return table[compindices]

def join_by_id(table1, table2, columnid1, columnid2, join_type="inner",
               conflict_suffixes=("_A", "_B"), idproc=None,
               additional_keys=[]):
    '''Joins two tables based on columns with different names.

    Table1 and table2 are the tables to be joined together. The column names
    that should be joined are the two columnids. Columnid1 will be the column
    name for the returned table. In case of conflicts, the
    conflict suffixes will be appended to the keys with conflicts. To merge
    conflicts instead of keeping them separate, add the column name to
    additional_keys.

    If the entries in the columns to be merged should be processed a certain
    way, the function that does the processing should be given in idfilter. For
    no processing, "None" should be passed instead.
    '''
    # There are some corner cases that should be dealt with in order for this
    # function to work *perfectly*
    tempcol1 = table1[columnid1]
    tempcol2 = table2[columnid2]

    # Process the columns if need be.
    if idproc is not None:
        tempcol1 = idproc(tempcol1)
        tempcol2 = idproc(tempcol2)

    # If columnid1 = columnid2, then we can go straight to a join. If not, then 
    # columnid2 needs to be renamed to columnid1. If table2[columnid1] exists, 
    # then we have a problem and an exception should be thrown.
    if columnid1 != columnid2:
        if columnid1 not in table2.colnames:
            table2[columnid1] = tempcol2
        else: 
            raise ValueError(
                "Column {0} already exists in second table.".format(columnid1))

    try:
        newtable = join(
            table1, table2, keys=[columnid1]+additional_keys, 
            join_type=join_type, table_names=list(conflict_suffixes), 
            uniq_col_name="{col_name}{table_name}")
    finally:
        # Clean up the new table.
        if columnid1 != columnid2:
            del(table2[columnid1])

    return newtable

def join_by_ra_dec(
    table1, table2, ra1="RA", dec1="DEC", ra2="RA", dec2="DEC", 
    ra1_unit=u.degree, dec1_unit=u.degree, ra2_unit=u.degree, dec2_unit=u.degree, 
    match_threshold=5*u.arcsec, join_type="inner", 
    conflict_suffixes=("_A", "_B")):
    '''Join two tables by RA and DEC.

    This function will essentially perform a join between tables using
    coordinates. The column names for the coordinates should be given in ra1,
    ra2, dec1, dec2. 

    In case of conflicts, the conflict_suffices will be used for columns in
    table1 and table2, respectively.
    '''
    # Instead of directly using RA/Dec, we'll set up a column that maps rows in
    # table 2 to rows in table2.
    match_column = generate_random_string(10)

    ra1_coords = table1[ra1]
    try:
        ra1_coords = ra1_coords.to(ra1_unit)
    except u.UnitConversionError:
        ra1_coords = ra1_coords * ra1_unit

    dec1_coords = table1[dec1]
    try:
        dec1_coords = dec1_coords.to(dec1_unit)
    except u.UnitConversionError:
        dec1_coords = dec1_coords * dec1_unit
    ra2_coords = table2[ra2]
    try:
        ra2_coords = ra2_coords.to(ra2_unit)
    except u.UnitConversionError:
        ra2_coords = ra2_coords * ra2_unit
    dec2_coords = table2[dec2]
    try:
        dec2_coords = dec2_coords.to(dec2_unit)
    except u.UnitConversionError:
        dec2_coords = dec2_coords * dec2_unit

    # This will cross-match the two catalogs to find the nearest matches.
    coords1 = SkyCoord(ra=ra1_coords, dec=dec1_coords)
    coords2 = SkyCoord(ra=ra2_coords, dec=dec2_coords)
    idx, d2d, d3d = coords1.match_to_catalog_sky(coords2)

    # We only count matches which are within the match threshold.
    matches = d2d < match_threshold
    matched_tbl1 = table1[matches]

    try:
        table2[match_column] = np.arange(len(table2))

        matched_tbl1[match_column] = table2[idx[matches]][match_column]

        newtable = join(
            matched_tbl1, table2, keys=match_column, 
            join_type=join_type, table_names=list(conflict_suffixes),
            uniq_col_name="{col_name}{table_name}")

    finally:
        del(table2[match_column])

    del(newtable[match_column])
    # Want to inherit table1 column naming.
    # This will require deleting the table2 coordinates from the new table.
    del(newtable[ra2])
    del(newtable[dec2])

    return newtable

def generate_random_string(length):
    '''Generate a random string with the given length.'''
    return "".join([random.choice(string.ascii_letters) for _ in
                    range(length)])

def get_complement_indices(initindices, tablelength):
    '''Returns the indices corresponding to rows not in partialtable.
    
    This function essenially creates indices which correspond to the rows in
    totaltable rows not in partialtable.
    '''
    compmask = np.ones(tablelength, np.bool)
    compmask[initindices] = 0
    return np.where(compmask)

def get_complement_table(partialtable, totaltable, compcolumn):
    '''Returns a subtable of total table without rows in partialtable.

    This is kinda like an operation to create a table which when stacked with
    partialtable and sorted by compcolumn, will create totaltable.
    '''
    partialindices = astropy_table_indices(totaltable, compcolumn,
                                                partialtable[compcolumn])
    compmask = get_complement_indices(partialindices, len(totaltable))
    comp_sample = totaltable[compmask]
    return comp_sample

def split_table_by_value(table, column, splitvalue):
    '''Bifurcates a table in two.

    This function splits a table based on the values in column and returns two
    tables in a 2-tuple. Values less than splitvalue are in the first tuple.
    Values greater than splitvalue are in the second.
    '''
    lowentries = table[np.where(table[column] < splitvalue)]
    highentries = table[np.where(table[column] >= splitvalue)]

    return lowentries, highentries
        
def first_row_in_group(tablegroup):
    '''Iterates through groups and selects the first row from each group.

    This is good for tables where there are multiple entries for each grouping,
    but the first row in the table is the preferable one. Such a thing occurs
    with the Catalog of Active Binary Systems (III).
    '''
    rowholder = []
    for group in tablegroup.groups:
        rowholder.append(group[0])
    filteredtable = Table(rows=rowholder, names=tablegroup.colnames)
    return filteredtable

###############################################################################
# Astroquery Catalog #
###############################################################################

def Vizier_cached_table(tblpath, tablecode):
    '''Read a table from disk, querying Vizier if needed.

    For large tables which can be automatically queried from Vizier, but take a
    long time to download, this function will download the queried table into
    tblpath, and then read from it for all following times.

    The tablecode is the code (e.g. "J/A+A/512/A54/table8") uniquely
    identifying the desired table.'''

    try:
        tbl = Table.read(str(tblpath), format="ascii.ipac")
    except FileNotFoundError:
        Vizier.ROW_LIMIT = -1
        tbl = Vizier.get_catalogs(tablecode)[0]
        tbl.write(str(tblpath), format="ascii.ipac")

    return tbl

###############################################################################
# Spreadsheet help #
###############################################################################
def inspect_table_as_spreadsheet(table):
    '''Opens the table in Libreoffice.

    For cases where it would be much easier to look at data by analyzing it in
    a spreadsheet, this function will essentially take the table and load it
    into Libreoffice so that operations can be done on it.
    '''
    with tempfile.NamedTemporaryFile() as fp:
        table.write(fp.name, format="ascii.csv")
        libreargs = ["oocalc", fp.name]
        subprocess.run(libreargs)

def inspect_table_in_topcat(table):
    '''Opens the table in TOPCAT

    TOPCAT is a useful tool for inspecting tables that are suited to be written
    as FITS files. TOPCAT is actually much more extensible than we are using it
    for, but it's helpful for this purpose.
    '''
    with tempfile.NamedTemporaryFile() as fp:
        table.write(fp.name, format="fits", overwrite=True)
        topcatargs = ["/home/regulus/simonian/topcat/topcat", fp.name]
        subprocess.run(topcatargs)

###############################################################################
# Caching large data files #
###############################################################################

class memoized(object):
    '''Decorator. Cache's a function's return value each time it is called. If
    called later with the same arguments, the cached value is returned (not
    reevaluated).
    '''
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if not isinstance(args, collections.Hashable):
            # uncacheable. a list, for instance.
            # better to not cache than blow up
            print("Uncacheable")
            return self.func(*args)
        if args in self.cache:
            print("Cached")
            return self.cache[args]
        else:
            print("Putting into cache")
            value = self.func(*args)
            self.cache[args] = value
            return value
    def __repr__(self):
        '''Return the function's docstring.'''
        return self.func.__doc__
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)

###############################################################################
# Itertools help #
###############################################################################

def roundrobin(*iterables):
    '''roundrobin('ABC', 'D', 'EF') --> ADEBFC'''
    # Recipe cedited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

def take(n, iterable):
    '''Return first n items of the iterable as a list.'''
    return list(islice(iterable, n))
