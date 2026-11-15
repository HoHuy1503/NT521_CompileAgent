CC ?= gcc
CFLAGS ?= -O0 -fno-stack-protector -no-pie -Wall
LDFLAGS ?= -no-pie -Wl,-z,norelro

TARGET = vuln_demo

all: $(TARGET)

$(TARGET): vuln_demo.c
	$(CC) $(CFLAGS) -o $(TARGET) vuln_demo.c $(LDFLAGS)

test: $(TARGET)
	./$(TARGET) --test

clean:
	rm -f $(TARGET) *.o
