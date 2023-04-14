#run preprocessing
preprocessing:
	@python mountain_goat/preprocessing.py

#run clean
clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr mountain_goat-*.dist-info
	@rm -fr mountain_goat.egg-info
	@rm -f */*/*/.DS_Store

main:
	@python mountain_goat/main.py

clean_main: clean main
