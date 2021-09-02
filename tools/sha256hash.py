import sys
import hashlib

def calc_hash(file_name):
	return hashlib.sha256(file_name.encode('utf-8')).hexdigest()

def main():
	args = sys.argv[1:]
	for arg in args:
		print(calc_hash(str(arg)))

if __name__ == "__main__":
	main()
