clear

cd ./bin

echo "Compiling Source Files ... "
g++ -Wall -std=c++20 -o AES ../export/main.cpp ../src/Common/*.cpp ../src/AES/*.cpp ../src/Huffman/*.cpp 

cp AES ../

cd ..

chmod +x ./AES

./AES
