from scAI_SNP import test_mic

def test_test_mic(capsys):
	test_mic()
	captured = capsys.readouterr()
	assert captured.out == 'mic test one two three\n'
	assert captured.err == ''
