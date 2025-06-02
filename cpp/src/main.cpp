#include "aiya_jetson.h"
#include "logging.h"
#include <chrono>
#include <iostream>
#include <signal.h>
#include <thread>

bool signal_recieved = false;

void sig_handler(int sig_no) {
  if (sig_no == SIGINT) {
    std::cout << "received SIGINT\n";
    signal_recieved = true;
  }
}
int main(int argc, char **argv) {
  Log::SetLevel(Log::Level::INFO);

  // Register signal handler for SIGINT
  signal(SIGINT, sig_handler);

  AIYAApp app;
  app.Start();

  while (!signal_recieved) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }

  return 0;
}