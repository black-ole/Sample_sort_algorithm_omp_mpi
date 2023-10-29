#include<iostream>
#include<mpi.h>
#include<string>
#include<fstream>
#include<algorithm>
#include<vector>
#include<omp.h>
#include<cstring>
#include<cstdio>
#include<chrono>
#include<sys/stat.h>
#include"psort.h"

#define ull unsigned int
#define pll pair<ull,ull>
using namespace std;


int main(int argc, char *argv[]){

        MPI::Init(argc, argv);

        ull rank = MPI::COMM_WORLD.Get_rank();
        ull p = MPI::COMM_WORLD.Get_size();
		
        string inputfile = argv[1];
		inputfile = inputfile.substr(12, inputfile.size() - 12 + 1);
		
		struct stat results;
		ull size_of_file;
		if(stat(inputfile.c_str(), &results) == 0){
			size_of_file = results.st_size;
		}
		ull size_of_input = size_of_file/8; 
		
		fstream infile;
		infile.open(inputfile.c_str(), ios::binary | ios::in);
		
		ull seek_pos = 0;
		ull num_of_elements = size_of_input/p;
		
		if(size_of_input % p >= (rank + 1)){
			num_of_elements++;
			seek_pos = num_of_elements * rank;
		}else{
			seek_pos = (size_of_input % p) + num_of_elements * rank;
		}
		

		infile.seekg(8*seek_pos);
		pll* inputdata = new pll[num_of_elements];
		ull l = 0;
		for(ull i = 0; i < num_of_elements ; i++, l++){
		  
		  for(int k = 0; k < 2; k++){
			  ull num = 0;
			  ull x = 0;
			  for(int j = 0; j < 4; j++){
				infile.read((char*)&num , 1);
				x = (x << 8) | num;
			  }
			  if(k == 0){
			  	inputdata[l].first = x;
			  }else{
			  	inputdata[l].second = x;
			  }	  
		  }
		}
       infile.close();
       

        string temp = argv[3];
        int BUCKETS = stoi(temp.substr(4, temp.size() - 4 + 1));
		
        ParallelSort(inputdata, num_of_elements, BUCKETS);

        ull* pseudo_splitters = new ull[p];
        l = 0;
        for(ull i = 0; i < num_of_elements && l < (ull)p; i += num_of_elements/p ){
                pseudo_splitters[l++] = inputdata[i].first;
        }

        ull* splitters = new ull[p*p];

        MPI::COMM_WORLD.Gather(pseudo_splitters, p, MPI::INT, splitters, p , MPI::INT, 0);        

        MPI::COMM_WORLD.Barrier();


        ull* real_splitters = new ull[p-1];

        if(rank == 0){
                sort(splitters, splitters + p*p);

                for(ull i = 0; i < p - 1; i++){
                        real_splitters[i] = splitters[(i+1)*p];
                }
        }

        MPI::COMM_WORLD.Bcast(real_splitters, p - 1, MPI::INT, 0);
        MPI::COMM_WORLD.Barrier();
       
        ull* bucket_count = new ull[p];
        ull prev = 0, index;
        
        for(ull i = 0; i < p - 1; i++){

                index = lower_bound(inputdata, inputdata + num_of_elements, make_pair(real_splitters[i] + 1, (ull)0)) - inputdata;

                bucket_count[i] = index - prev;
                prev += bucket_count[i];
        }

        bucket_count[p-1] = num_of_elements - index;

        ull size_of_output ;

        for(ull i = 0 ; i < p ; i++){
                MPI::COMM_WORLD.Reduce(&bucket_count[i], &size_of_output , 1, MPI::INT, MPI::SUM, i);
        }

        MPI::COMM_WORLD.Barrier();

        pll* outputdata = new pll[size_of_output];

        ull offset = 0;
        for(ull i = 0; i < p; i++){

                MPI::COMM_WORLD.Isend(inputdata + offset, sizeof(pll)*bucket_count[i], MPI::BYTE, i, i);
                offset += bucket_count[i];
        }

        offset = 0;

        for(ull i = 0; i < p; i++){

                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                int count;
                MPI_Get_count(&status, MPI::BYTE, &count);
                MPI::COMM_WORLD.Recv(outputdata + offset, count, MPI::BYTE, status.MPI_SOURCE, status.MPI_TAG);
                offset += count/sizeof(pll);
        }

        MPI::COMM_WORLD.Barrier();

        ParallelSort(outputdata, size_of_output, BUCKETS);

        string outputfile = argv[2];
		outputfile = outputfile.substr(13, outputfile.size() - 13 + 1);
        
        
		
		int check[1];
		check[0] = 0;
        	
        if(rank != 0){
			MPI::COMM_WORLD.Recv(check, 1, MPI::INT, rank - 1, 0);
		}
       	
		FILE *my_file ;
		if(rank == 0){
			my_file = fopen(outputfile.c_str(), "w+");
			fclose(my_file);
		}
		my_file = fopen(outputfile.c_str(), "a");
		
		for(ull i = 0; i < size_of_output ; ++i){

			outputdata[i].first = __builtin_bswap32(outputdata[i].first);
			fwrite(&outputdata[i].first, sizeof(ull), 1, my_file);
			outputdata[i].second = __builtin_bswap32(outputdata[i].second);
			fwrite(&outputdata[i].second, sizeof(ull), 1, my_file);
		}
		
		fclose(my_file);
				
        if(rank != p-1){
			MPI::COMM_WORLD.Send(check, 1, MPI::INT, rank + 1, 0);
		}
        
        MPI::COMM_WORLD.Barrier();

        delete[] inputdata;
        delete[] pseudo_splitters;
        delete[] splitters;
        delete[] real_splitters;
        delete[] bucket_count;
        delete[] outputdata;

        MPI::Finalize();

        return 0;
}
