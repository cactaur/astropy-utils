import pytest
from astropy.table import Table

@pytest.fixture
def diverse_table():
    tbl = Table([
        ["0", "1", "2", "3", "4", "5"], [0, 1, 2, 3, 4, 5], 
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], names=(
            "String Column", "Int Column", "Float Column"))
    return tbl

@pytest.fixture
def diverse_masked_table():
    tbl = Table([
        ["0", "1", "2", "3", "4", "5"], [0, 1, 2, 3, 4, 5], 
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], names=(
            "String Column", "Int Column", "Float Column"), masked=True)
    tbl["String Column"].mask[3] = True
    tbl["Int Column"].mask[4] = True
    tbl["Float Column"].mask[5] = True
    return tbl

class TestIndexing:
    def test_
