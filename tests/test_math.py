import random

from scAI_SNP.math import center, div_int, cmd_center


def test_center():
    assert center(1, 2) == -1.00
    assert center(0.1, 3) == -2.90
    a, b = random.random(), random.random()
    assert center(a, b) == round(a - b, 2)


def test_div_int():
    assert div_int(3, 2) == 1
    assert div_int(3, 1.6) == 1
    a, b = random.random(), random.random()
    assert div_int(a, b) == a // b


def test_cmd_center(capsys):
    cmd_center(['1', '3'])
    cmd_center(['1', '2.2'])
    captured = capsys.readouterr()
    assert captured.out == '-2.0\n-1.2\n'
    assert captured.err == ''
