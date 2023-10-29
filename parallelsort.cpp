#include <iostream>
#include "psort.h"
#include <omp.h>
#include <algorithm>

#define ull unsigned int
#define pll pair<ull,ull>

using namespace std;

void update_ele_cnt(ull *realsplit , int n, ull *elecnt , pll *data, ull l, ull r){

	for(int i = 0 ; i < n ; i++){
		
		ull idx = lower_bound(data + l, data + r + 1, make_pair(realsplit[i] + 1,(ull)0)) - data;
		#pragma omp atomic
			elecnt[i] += idx - l;		
	}
}

void copydata(pll *copy, pll *data, ull l, ull r){

	for(ull i = l ; i <= r; i++){
		data[i] = copy[i];
	}
	return;
}

void ParallelSort(pll *data, ull n, int p){
	
	ull *pseudosplit = new ull[p*p];
	ull *adj = new ull[p+1];
	adj[0] = 0;
	for(ull i= 1; i <= (ull)p; i++){
		if(i-1 < n % p){
			adj[i] = (n/p) + 1;
		}else{
			adj[i] = (n/p);
		}
		adj[i] += adj[i-1];
	}
	
	#pragma omp parallel for num_threads(p)
	for(ull i = 1; i <= (ull)p; i++ ){
		
			stable_sort(data + adj[i-1], data + adj[i]);
			
			ull tt = 0;
			for(ull k = adj[i-1]; k <= adj[i] - 1; k += (n/p)/p){
				
				pseudosplit[(i-1)*p + tt] = data[k].first;
				
				tt++;
				if(tt % p == 0){
					break;
				}
 			}		
	}
	

	stable_sort(pseudosplit, pseudosplit + p*p );

	ull *realsplit = new ull[p-1];
	for(int i = 0; i < p - 1; i++){
		realsplit[i] = pseudosplit[(i+1)*p];
	} 

	ull *elecnt = new ull[p];
	for(int i = 0; i < p ; i++){
		elecnt[i] = 0;
	}
	
	#pragma omp parallel for num_threads(p)
	for(ull i = 1; i <= (ull)p ; i++){
		update_ele_cnt(realsplit, p - 1, elecnt, data , adj[i-1] , adj[i] - 1);
	}

	elecnt[p-1] = n;
	for(int i = p - 1; i >= 1 ; i--){
		elecnt[i] -= elecnt[i-1];
	}	
	
	ull* pos = new ull[p+1];
	pos[0]= 0;
	for(int i = 1; i < p+1; i++){
		pos[i] = elecnt[i-1] + pos[i-1];
	}

	ull** start_pos = new ull*[p];
	for(int i = 0; i < p; i++){
		start_pos[i] = new ull[p];
	}
	
	#pragma omp parallel num_threads(p)
	{
		int curr_bucket = omp_get_thread_num();
		ull l = adj[curr_bucket], r = adj[curr_bucket + 1] - 1; 
		for(int i = 0; i < p-1; i++){
			ull idx = lower_bound(data + l, data + r + 1, make_pair(realsplit[i] + 1, (ull)0)) - data;
			start_pos[curr_bucket][i] = idx - l;
		}
		start_pos[curr_bucket][p-1] = r - l + 1;
	}
	

	pll*copy = new pll[n];

	#pragma omp parallel num_threads(p)
	{
		
		ull to_bucket = omp_get_thread_num();
		
		ull base_copy = pos[to_bucket] , from_bucket = 0;
		
		while(from_bucket < (ull) p){
			ull offset_data = 0, till_pos = start_pos[from_bucket][to_bucket], base_data = adj[from_bucket];
			
			if(to_bucket != 0){
				offset_data = start_pos[from_bucket][to_bucket - 1];
			}
			
			for(ull i = offset_data ; i < till_pos; i++, base_copy++){
				copy[base_copy] = data[base_data + i];	
			}
			
			from_bucket++;
		}
		
	}

	#pragma omp parallel for num_threads(p)
	for(int i = 1; i <= p; i++){
		copydata(copy,data,adj[i-1],adj[i]-1);
	}
	
	#pragma omp parallel for num_threads(p)
	for(int i = 1; i <= p; i++){
		stable_sort(data + pos[i-1], data + pos[i]); 
	}	
	

	delete[] pseudosplit;
	delete[] adj;
	delete[] realsplit;
	delete[] elecnt;
	for(int i = 0; i < p; i++){
		delete start_pos[i];
	}
	delete[] start_pos;
	delete[] copy;

	return;
}

