
int main(){
    int a = 3;
    int *i = a;
    int *j = &a;
    int **k = &j;

    int ***l = &k;
    int **m = a;
    int ***n = a;

    int o = k;
    int **p = k;

    **k = 10;


    printf(" a : %d , %p\n", a, &a);
    printf(" i : %d , %p\n", i, &i);
    printf(" j : %d , %p, %p\n", *j, j, &j);
    printf(" k : %d , %p, %p, %p\n", **k, *k, k, &k);
    printf(" l : %d , %p, %p, %p, %p\n", ***l, **l, *l, l, &l);
    printf(" o : %p , %p\n", o, &o);
    printf(" p : %d , %p\n", **p, &p);
    printf("================================\n");
    m=4;
    a=5;
    printf(" a : %d , %p\n", a, &a);
    printf(" m : %d , %p\n",  m, &m);
    printf(" n : %d , %p\n",  n, &n);
    // printf("%d -> %p, %p\n", a, &a, &a);

    // memset(img, 0, N * C * H * W  * sizeof(unsigned char));

    // free(img);

    return 0;
}