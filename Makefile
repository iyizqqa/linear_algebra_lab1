all:
	g++ main.cpp matrix.cpp methods.cpp experiments.cpp -o lab1

run:
	./lab1

clean:
	rm -f lab1
