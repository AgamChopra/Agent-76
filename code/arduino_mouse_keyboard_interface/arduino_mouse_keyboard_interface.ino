#include <Mouse.h>
#include <Keyboard.h>

String cmd;
int msg_i = 0;
int mouse_x = 0;
int mouse_y = 0;
int idx = 0;
int ix = 0;

const int NUM_COMMANDS = 16;
const char commandChars[NUM_COMMANDS - 4] = " qwerasdfgx";

void setup() {
  Serial.begin(9600);
  Mouse.begin();
  Keyboard.begin();
}

void movement(String& str) {
  idx = str.indexOf(',');
  mouse_x = str.substring(0, idx).toInt();
  mouse_y = str.substring(idx + 1).toInt();
  Mouse.move(mouse_x, mouse_y, 0);
}

void command(String& str, int i) {
  if (i == 0) movement(str);
  else{
    if (str == "1") {
      if (i == 1 || i == 2) Mouse.press(i == 1 ? MOUSE_LEFT : MOUSE_RIGHT);
      else if (i == 3 || i == 4) Keyboard.press(i == 3 ? KEY_LEFT_SHIFT : KEY_LEFT_CTRL);
      else Keyboard.press(commandChars[i - 5]);
    }
    else {
      if (i == 1 || i == 2) Mouse.release(i == 1 ? MOUSE_LEFT : MOUSE_RIGHT);
      else if (i == 3 || i == 4) Keyboard.release(i == 3 ? KEY_LEFT_SHIFT : KEY_LEFT_CTRL);
      else Keyboard.release(commandChars[i - 5]);
    }
  }
}

void loop() {
  if (Serial.available() > 0) {
    String msg = Serial.readStringUntil('\n');
    msg_i = 0;
    ix = 0;
    while (ix < NUM_COMMANDS) {
      idx = msg.indexOf(' ');
      cmd = msg.substring(0, idx);
      msg = msg.substring(idx + 1);
      command(cmd, ix);
      ix++;
    }
    delayMicroseconds(0.5);
  }
}
