#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pthread.h>
#include "thpool.h"

#include <unistd.h>

#define THREAD_MAX 10

struct arg_struct {
    unsigned char *img;
    unsigned char *mask;
    unsigned long N;
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



// void Cargmax(void *img_array, unsigned long N ){
//     unsigned long C = 21, H = 360, W = 640;
//     unsigned char *out = malloc(N * H * W * sizeof(unsigned char) );

//     struct arg_struct args[THREAD_MAX];
//     for(int i = 0 ; i < THREAD_MAX ; i++){
//         args[i].img = img_array;
//         args[i].mask = out;
//         args[i].N = N;
//         args[i].thread_id = i;
//     }

//     threadpool thpool = thpool_init(THREAD_MAX);

//     for (int i = 0; i < THREAD_MAX; i++) { //create multiple threads
//         thpool_add_work(thpool, (void*)&argmax3D_thread, (void*) &args[i]);
//     };

//     thpool_wait(thpool);
//     thpool_destroy(thpool);
// }


void Cargmax(void *img_array, void *mask_array, unsigned long N ){
    unsigned long C = 21, H = 360, W = 640;
    // unsigned char *out = malloc(N * H * W * sizeof(unsigned char) );

    struct arg_struct args[THREAD_MAX];
    for(int i = 0 ; i < THREAD_MAX ; i++){
        args[i].img = img_array;
        args[i].mask = mask_array;
        args[i].N = N;
        args[i].thread_id = i;
    }

    threadpool thpool = thpool_init(THREAD_MAX);

    for (int i = 0; i < THREAD_MAX; i++) { //create multiple threads
        thpool_add_work(thpool, (void*)&argmax3D_thread, (void*) &args[i]);
    };

    thpool_wait(thpool);
    thpool_destroy(thpool);
}


// clear && gcc -shared -fPIC argmax_API.c thpool.c -D THPOOL_DEBUG -pthread  -o argmax.so
