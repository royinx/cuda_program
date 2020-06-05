#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include "thpool.h"

#include <time.h>

#include <unistd.h>

// unsigned char THREAD_MAX;

#define THREAD_MAX 24

#define IMG_BATCH (unsigned long) 2500

int total_img = 0;
struct arg_struct {
    unsigned char *img;
    unsigned char *mask;
    unsigned long N;
    unsigned char thread_id;
};

struct dims {
    unsigned char *img;
    unsigned long N;
    unsigned long C;
    unsigned long H;
    unsigned long W;
    unsigned char thread_id;
};

void* argmax3D_thread(struct arg_struct *arg){
    unsigned char thread_id = (*arg).thread_id;
    unsigned char *img = (*arg).img;
    unsigned char *mask = (*arg).mask;
    unsigned long N = (*arg).N;
    // printf("thread_id : %d , n: %ld \n", thread_id, N);

    // unsigned long N = 500;
    int C = 21, H = 360, W = 640;
    unsigned long in_index, out_index;
    int ign_cls[14] = {0,1,2,3,4,5,9,10,13,14,15,16,17,18};
    int pixel;
    long n, h, w, c, ign;

    for(n=thread_id ; n<N ; n+=THREAD_MAX){
        for(h=0 ; h<H ; h++){
            for(w=0 ; w<W ; w++){
                pixel = 0;
                for(c=0; c<C ; c++){
                    in_index = n*C*H*W + c*H*W + h*W + w;
                    out_index = n*H*W + h*W + w;

                    if(pixel <= img[in_index]){
                        pixel = img[in_index];
                        mask[out_index] = c;
                    }
                }
                for(ign =0; ign<14 ; ign++){
                    if (mask[out_index]==ign_cls[ign]){
                        mask[out_index] = 0;
                    }
                }
            }
        }
        // printf("thread_id : %d , n: %d , cls: %d \n", thread_id, n, mask[out_index]);
        // total_img++;
    }
    return NULL;
}


void* init_matrix(struct dims *dim){
    // printf("thread_id : %d \n", dim->thread_id);
    unsigned char *img = (*dim).img;
    unsigned long N = (*dim).N;
    unsigned long C = (*dim).C;
    unsigned long H = (*dim).H;
    unsigned long W = (*dim).W;
    unsigned char thread_id = (*dim).thread_id;

    unsigned char value = 0;
    unsigned long index;
    long n, h, w, c, ign;
    for(n=thread_id ; n<N ; n+=THREAD_MAX){
        // printf("thread_id : %d , n: %ld, N: %ld \n", thread_id, n, N);
        for(c=0 ; c<C ; c++){
            for(h=0 ; h<H ; h++){
                for(w=0 ; w<W ; w++){
                    index = n*C*H*W + c*H*W + h*W + w;
                    img[index] = value;
                    value += 1;
                }
            }
        }
        total_img++;
    }
    return NULL;
}
// void* init_matrix2(struct dims *dim){
//     int thread_id  = (*dim).thread_id;
//     unsigned char *img = (*dim).img;
//     unsigned long N = (*dim).N;
//     unsigned long C = (*dim).C;
//     unsigned long H = (*dim).H;
//     unsigned long W = (*dim).W;

//     unsigned char value = 0;
//     unsigned long index;
//     for(int n=0 ; n<N ; n++){
//         for(int c=0 ; c<C ; c++){
//             for(int h=0 ; h<H ; h++){
//                 for(int w=0 ; w<W ; w++){
//                     index = n*C*H*W + c*H*W + h*W + w;
//                     img[index] = value;
//                     value += 1;
//                 }
//             }
//         }
//     }
//     return NULL;
// }

 
int main(){
    // const long THREAD_MAX = sysconf(_SC_NPROCESSORS_ONLN)*0.75;
    unsigned long N = IMG_BATCH, C = 21, H = 360, W = 640;
    unsigned char *img = malloc(N * C * H * W * sizeof(unsigned char) );
    unsigned char *out = malloc(N * H * W * sizeof(unsigned char) );
    
    if( img == NULL | out == NULL ) {
        fprintf(stderr, "Error: unable to allocate required memory\n");
        return 1;
    }


    struct timespec start, end;
    u_int64_t delta_us ;
    printf("====================== Init Array ======================\n");

    struct dims dim[THREAD_MAX];
    for(int i = 0 ; i < THREAD_MAX ; i++){
        dim[i].thread_id = i;
        dim[i].img = img;
        dim[i].N = N;
        dim[i].C = C;
        dim[i].H = H;
        dim[i].W = W;
    }


    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    threadpool thpool = thpool_init(THREAD_MAX);
    for (int i = 0; i < THREAD_MAX; i++) { //create multiple threads
        thpool_add_work(thpool, (void*)&init_matrix, (void*) &dim[i]);
    };   
    thpool_wait(thpool);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("%ld us\n ", delta_us);
    printf("%ld ms\n", delta_us/1000);


    printf("====================== Calc Array ======================\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    struct arg_struct args[THREAD_MAX];
    for(int i = 0 ; i < THREAD_MAX ; i++){
        args[i].img = img;
        args[i].mask = out;
        args[i].N = N;
        args[i].thread_id = i;
    }


    for (int i = 0; i < THREAD_MAX; i++) { //create multiple threads
        thpool_add_work(thpool, (void*)&argmax3D_thread, (void*) &args[i]);
    };
    thpool_wait(thpool);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    delta_us = (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
    printf("%ld us\n ", delta_us);
    printf("%ld ms\n", delta_us/1000);

    printf("%d \n", total_img);


    thpool_destroy(thpool);

    free(img);
    free(out);

    return 0;
}