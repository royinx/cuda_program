# (Multi-threading) Argmax in C

```bash
git submodule update --init --remote --recursive

gcc -g argmax.c  thpool.c -D THPOOL_DEBUG -pthread -o argmax.o && time ./argmax.o
```