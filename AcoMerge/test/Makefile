.PHONY: all

all:
	#gcc -o aco_cpu ../src/*.c ./aco_cpu.c -lm -lpthread -O3 -mavx2 -g -I/usr/local/cuda/include -lOpenCL
	#gcc -o aco_cpu_edges_type ../src/*.c ./aco_cpu_edges_type.c -lm -lpthread -O3 -mavx2 -g -I/usr/local/cuda/include -lOpenCL
	#gcc -o aco_cpu_lstm ../src/*.c ./aco_cpu_lstm.c -lm -lpthread -O3 -mavx2 -g -I/usr/local/cuda/include -lOpenCL
	#gcc -o aco_gpu_lstm ../src/*.c ./aco_gpu_lstm.c -lm -lpthread -O3 -mavx2 -g -I/usr/local/cuda/include -lOpenCL
	#gcc -o aco_gpu ../src/*.c ./aco_gpu.c -lm -lpthread -O3 -mavx2 -g -I/usr/include /usr/lib/x86_64-linux-gnu/libOpenCL.so.1
	#gcc -o pso_cpu ../src/*.c ./pso_cpu.c -lm -lpthread -O3 -mavx2 -g -I/usr/local/cuda/include -lOpenCL
	#gcc -o aco_edge_popup_cpu ../src/*.c ./aco_edge_popup_cpu.c -lm -lpthread -O3 -mavx2 -g -I/usr/local/cuda/include -lOpenCL
	gcc -o aco_edge_popup_cpu_cl ../src/*.c ./aco_edge_popup_cpu_cl.c -lm -lpthread -O3 -mavx2 -g -I/usr/local/cuda/include -lOpenCL
	#gcc -o aco_edge_popup_cpu_lstm ../src/*.c ./aco_edge_popup_cpu_lstm.c -lm -lpthread -O3 -mavx2 -g -I/usr/local/cuda/include -lOpenCL
	#gcc -o louvain ../src/*.c ./louvain.c -lm -lpthread -O3 -mavx2 -g -I/usr/local/cuda/include -lOpenCL
