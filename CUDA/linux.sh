clear

rm AES
cd ./bin
rm *

echo "Compiling Source Files ... "
nvcc -o AES -ccbin "/opt/cuda/bin" --std=c++20 -lcudart -I/opt/cuda/include ../src/AES/*.cu ../src/Huffman/encoder/*.cu ../src/Huffman/decoder/*.cu ../src/Huffman/encoder/*.cpp ../src/Huffman/decoder/*.cpp ../src/Common/*.cpp ../export/main.cu

cp AES ../
cd ..
chmod +x ./AES
./AES
