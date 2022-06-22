nvcc reduce.cu -c -o reduce.o -gencode=arch=compute_75,code=sm_75
g++ -mavx -o out reduce.cpp reduce.o -L/usr/local/cuda/lib64 -lcuda -lcudart -O3
# -Wl,--no-as-needed,-lprofiler,--as-needed

export CPUPROFILE=./profile.prof
