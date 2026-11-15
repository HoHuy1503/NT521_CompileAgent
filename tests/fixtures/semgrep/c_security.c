#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <syslog.h>
#include <unistd.h>

void test_unsafe_string_copy(const char *input, va_list ap) {
    char buf[64];
    char out[128];

    // ruleid: c-unsafe-string-copy
    strcpy(buf, input);

    // ruleid: c-unsafe-string-copy
    strcat(buf, input);

    // ruleid: c-unsafe-string-copy
    sprintf(out, "val=%s", input);

    // ruleid: c-unsafe-string-copy
    vsprintf(out, "fmt=%s", ap);

    // ruleid: c-unsafe-string-copy
    gets(buf);

    // ok: c-unsafe-string-copy
    snprintf(buf, sizeof(buf), "%s", input);

    // ok: c-unsafe-string-copy
    fgets(buf, sizeof(buf), stdin);
}

void test_command_injection(const char *arg) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "echo %s", arg);

    // ruleid: c-command-injection
    system(cmd);

    // ruleid: c-command-injection
    popen(cmd, "r");

    // ok: c-command-injection
    system("uname -a");

    // ok: c-command-injection
    FILE *fp = popen("ls -la", "r");
    if (fp) pclose(fp);
}

void test_format_string(const char *user_msg, int x) {
    // ruleid: c-format-string-vuln
    printf(user_msg);

    // ruleid: c-format-string-vuln
    fprintf(stderr, user_msg);

    // ruleid: c-format-string-vuln
    syslog(LOG_INFO, user_msg);

    // ok: c-format-string-vuln
    printf("%s", user_msg);

    // ok: c-format-string-vuln
    printf("value=%d\n", x);

    // ok: c-format-string-vuln
    fprintf(stderr, "error: %s\n", user_msg);

    // ok: c-format-string-vuln
    syslog(LOG_INFO, "msg: %s", user_msg);
}

void test_integer_overflow_alloc(size_t width, size_t height, void *old_ptr) {
    // ruleid: c-integer-overflow-alloc
    void *img = malloc(width * height);

    // ruleid: c-integer-overflow-alloc
    void *resized = realloc(old_ptr, width * height);

    // ok: c-integer-overflow-alloc
    void *safe_img = calloc(width, height);

    free(img);
    free(resized);
    free(safe_img);
}

void test_insecure_temp_file(void) {
    char tmpl1[] = "/tmp/app_XXXXXX";
    char tmpl2[] = "/tmp/app_XXXXXX";

    // ruleid: c-insecure-temp-file
    char *p1 = mktemp(tmpl1);
    (void)p1;

    // ruleid: c-insecure-temp-file
    char *p2 = tmpnam(NULL);
    (void)p2;

    // ruleid: c-insecure-temp-file
    char *p3 = tempnam("/tmp", "pre");
    free(p3);

    // ok: c-insecure-temp-file
    int fd = mkstemp(tmpl2);
    if (fd >= 0) close(fd);
}
