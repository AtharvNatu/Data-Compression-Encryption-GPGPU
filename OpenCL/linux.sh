clear

cd ./bin

echo "Compiling Source Files ... "
g++ -Wall -fPIC -std=c++20 -I "/opt/rocm-5.7.1/include/" -c ../export/lib.cpp ../src/Common/*.cpp ../src/AES/*.cpp ../src/Huffman/*.cpp ../src/CLFW/*.cpp

# echo "Linking Object Files ... "
# g++ -o AES *.o -L "/opt/rocm-5.7.1/lib/" -lOpenCL -lm

echo "Creating Shared Library ..."
g++ -shared -o libDCAEUG-OpenCL.so *.o -lOpenCL -lm

# cp Main ../
cp libDCAEUG-OpenCL.so ../

echo "Done ..." 

cd ..