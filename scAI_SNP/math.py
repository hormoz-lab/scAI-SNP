from returns import returns

@returns(int)
def div_int(x, y):
    return x / y

def center(x, y):
	return round(x - y, 2)

def cmd_center(args=None):
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('x', type=float)
	parser.add_argument('y', type=float)
	parsed_args = parser.parse_args(args)
	print(center(parsed_args.x, parsed_args.y))
