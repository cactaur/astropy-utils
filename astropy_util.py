"""A set of helper functions to work with the astropy module."""
import functools
import random
import string
import tempfile
import subprocess
import collections
from itertools import cycle, islice, chain, combinations

import scipy
import numpy as np
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
from astropy import units as u
#from astroquery.vizier import Vizier

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
        return np.array(index, dtype=np.bool)

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
    compindices = get_complement_indices(subindices, len(table))
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

    # Process the columns if need be.
    if idproc is not None:
        # I want to duplicate the data so it won't be lost. And by keeping it
        # in the table, it will be preserved when it is joined.
        origcol1 = table1[columnid1]
        origcol2 = table2[columnid2]
        randomcol1 = generate_random_string(10)
        randomcol2 = generate_random_string(10)
        table1.rename_column(columnid1, randomcol1)
        table2.rename_column(columnid2, randomcol2)
        table1[columnid1] = idproc(origcol1)
        table2[columnid2] = idproc(origcol2)

    # If columnid1 = columnid2, then we can go straight to a join. If not, then 
    # columnid2 needs to be renamed to columnid1. If table2[columnid1] exists, 
    # then we have a problem and an exception should be thrown.
    if columnid1 != columnid2:
        if columnid1 not in table2.colnames:
            table2[columnid1] = table2[columnid2]
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
        if idproc is not None:
            del(table1[columnid1])
            del(table2[columnid2])
            del(newtable[randomcol1])
            del(newtable[randomcol2])
            table1.rename_column(randomcol1, columnid1)
            table2.rename_column(randomcol2, columnid2)

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
    try:
        del(newtable[ra2])
    except KeyError:
        # This occurs when ra1=ra2.
        assert ra1==ra2
        newtable.rename_column(ra1+conflict_suffixes[0], ra1)
        del(newtable[ra2+conflict_suffixes[1]])

    try:
        del(newtable[dec2])
    except KeyError:
        assert dec1==dec2
        newtable.rename_column(dec1+conflict_suffixes[0], dec1)
        del(newtable[dec2+conflict_suffixes[1]])

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
        try:
            subprocess.run(libreargs)
        except FileNotFoundError:
            libreargs[0] = "localc"
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

def shortcut_file(filename, format="fits"):
    ''' Return a decorator that both caches the result and saves it to a file.

    This decorator should be used for commonly used snippets and combinations
    of tables that are small enough to be read in quickly, and processed enough
    that generating them from scratch is time-intensive.
    '''

    class Memorize(object):
        '''
        A function decorated with @memorize caches its return value every time
        it is called. If the function is called later with the same arguments,
        the cached value is returned (the function is not reevaluated). The
        cache is stored in the filename provided in shortcut_file for reuse in
        future executions. If the function corresponding to this decorated has
        been updated, make sure to change the object at the given filename.
        '''
        def __init__(self, func):
            self.func = func
            self.filename = filename
            self.table = None

        def __call__(self, *args):
            if self.table is None:
                try:
                    self.read_cache()
                except FileNotFoundError:
                    value = self.func(*args)
                    self.table = value
                    self.save_cache()

            return self.table

        def read_cache(self):
            '''
            Read the table in from the given location. This will take the
            format given in the shortcut_file command.
            '''
            self.table = Table.read(self.filename, format=format)

        def save_cache(self):
            '''
            Save the table into the given filename using the given format.
            '''
            try:
                self.table.write(self.filename, format=format)
            except FileNotFoundError:
                self.filename.parent.mkdir(parents=True)
                self.table.write(self.filename, format=format)

        def __repr__(self):
            ''' Return the function's docstring. '''
            return self.func.__doc__

        def __get__(self, obj, objtype):
            ''' Support instance methods. '''
            return functools.partial(self.__call__, obj)

    return Memorize

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

def flatten(listOfLists):
    "Flatten one level of nesting"
    return chain.from_iterable(listOfLists)

def random_permutation(iterable, r=None):
    """Random selection from itertools.product(*args, **kwds)"""
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def consume(iterator, n):
    "Advance the iterator n-steps ahead. If n is none, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)

###############################################################################
# Binary confidence intervals #
###############################################################################

def poisson_upper(n, sigma):
    '''Return the Poisson upper limit of the confidence interval.
    
    This is the upper limit for a given number of successes n, and the width of
    the confidence interval is given in sigmas.'''
    up = (n+1)*(1 - 1/9/(n+1) + sigma/3/np.sqrt(n+1))**3
    return up

def poisson_lower(n, sigma):
    '''Return the Poisson lower limit of the confidence interval.

    This is the lower limit for a given number of successes n, and the width of
    the confidence interval is given in sigmas. This formula is from Gehrels
    (1986) and contains tuned parameters.'''
    betas = {1.0: 0.0, 2.0: 0.062, 3.0:0.222}
    gammas = {1.0: 0.0, 2.0: -2.19, 3.0: -1.85}

    low = n * (1 - 1/9/n - sigma/3/np.sqrt(n) + betas[sigma]*n**gammas[sigma])**3
    return low

def binomial_upper(n1, n, sigma=1):
    '''The upper limit of the one-sigma binomial probability.

    This is the upper limit for a given number of successes n1 out of n trials.
    This is a numerically exact solution to the value.'''
    if sigma <= 0:
        raise ValueError("The probability needs to be positive.")
    cl = -scipy.special.erf(-sigma)
    ul = np.where(n1 != n, scipy.special.betaincinv(n1+1, n-n1, cl), 1)

    return ul

def binomial_lower(n1, n, sigma=1):
    '''The lower limit of the one-sigma binomial probability.

    This is the lower limit for a given number of successes n1 out of n trials.
    This provides a numerically exact solution to the value.'''
    ll = 1 - binomial_upper(n-n1, n, sigma=sigma)
    return ll



############################################################################
# Numpy help #
###############################################################################

def slicer_vectorized(arr, strindices):
    '''Extract the substring at strindices from an array.

    Given a string array arr, extract the substring elementwise corresponding
    to the indices in strindices.'''
    arr = np.array(arr, dtype=np.unicode_)
    indexarr = np.array(strindices, dtype=np.int_)
    temparr = arr.view('U1').reshape(len(arr), -1)[:,strindices]
    return np.fromstring(temparr.tostring(), dtype='U'+str(len(indexarr)))

###############################################################################
# Matplotlib Boundaries #
###############################################################################

def round_bound(lowbounds, upbounds, round_interval):
    '''Return a lower and upper bound within the given rounding interval.
    
    Generally the bounds should be the value plus or minus the error.
    
    Round-interval should be the width of the tick marks.'''

    minbound, maxbound = np.min(lowbounds), np.max(upbounds)

    lowlim = (minbound // round_interval) * round_interval
    highlim = ((maxbound // round_interval) + 1) * round_interval

    return lowlim, highlim

def adjust_axes(ax, lowx, highx, lowy, highy, xdiff, ydiff):
    '''Adjust the given axes to ensure all data fits within them.

    Ensure that the given matplotlib axes can accomodate both the new x and y
    limits provided in this function, as well as the internal x and y limits. 

    The tick intervals for x and y should be given in xdiff and ydiff.'''
    min_x, max_x = round_bound(lowx, highx, xdiff)
    min_y, max_y = round_bound(lowy, highy, ydiff)
    prev_xmin, prev_xmax = ax.get_xlim()
    prev_ymin, prev_ymax = ax.get_ylim()
    min_x = min(min_x, prev_xmin)
    max_x = max(max_x, prev_xmax)
    min_y = min(min_y, prev_ymin)
    max_y = max(max_y, prev_ymax)

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
