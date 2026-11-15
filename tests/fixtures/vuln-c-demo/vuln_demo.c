#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int unsafe_format_log(const char *msg) {
    int unused_flag = 1;
    (void)unused_flag;
    printf(msg);
    printf("\n");
    return 0;
}

static void copy_greeting(const char *name) {
    char buf[32];
    char banner[64];
    strcpy(buf, name);
    sprintf(banner, "Hello, %s", buf);
    puts(banner);
}

static int *alloc_matrix(size_t rows, size_t cols) {
    int *ptr = (int *)malloc(rows * cols);
    if (!ptr) {
        return NULL;
    }
    ptr[0] = 42;
    return ptr;
}

static int trigger_ubsan_signed_overflow(int a) {
    return a + 100;
}

static void trigger_asan_heap_oob(void) {
    char *heap = (char *)malloc(8);
    if (!heap) {
        return;
    }
    memset(heap, 'A', 8);
    heap[8] = '\0';
    printf("oob char=%d\n", (int)heap[8]);
    free(heap);
}

static void trigger_cppcheck_oob(int cond) {
    int arr[4] = {1, 2, 3, 4};
    if (cond == 9999) {
        char *leaked = (char *)malloc(32);
        if (leaked) {
            leaked[0] = 'z';
        }
        arr[4] = 99; /* static out-of-bounds + memleak for Cppcheck */
    }
    printf("arr[0]=%d\n", arr[0]);
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        puts("vuln_demo utility (use --test to run self-test)");
        return 0;
    }

    if (argc > 1 && strcmp(argv[1], "--test") == 0) {
        copy_greeting("AutoCompiler");
        unsafe_format_log("Running self-test");
        int *m = alloc_matrix(4, 4 * sizeof(int));
        free(m);
        volatile int maxv = INT_MAX;
        int ov = trigger_ubsan_signed_overflow(maxv);
        printf("overflow=%d\n", ov);
        trigger_asan_heap_oob();
        trigger_cppcheck_oob(argc);
        return 0;
    }

    copy_greeting(argc > 1 ? argv[1] : "world");
    return 0;
}
