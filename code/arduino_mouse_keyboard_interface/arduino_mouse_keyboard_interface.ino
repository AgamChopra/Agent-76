#include <Mouse.h>
#include <Keyboard.h>

String msgs[16];
int msg_i = 0;
int mouse_x = 0;
int mouse_y = 0;
int idx = 0;
int ix = 0;


void setup() {
  Serial.begin(9600);  
  Mouse.begin();
  Keyboard.begin();
}


void movement(String& str){
  Serial.print("..MOUSE MOVE..");
  idx = str.indexOf(',');
  mouse_x = str.substring(0, idx).toInt();
  mouse_y = str.substring(idx+1).toInt();
  Serial.print(mouse_x);
  Serial.print(mouse_y);
  Mouse.move(mouse_x, mouse_y, 0);
}


bool str2bool(String& str){
  if (str == "1" or str == "True"){
    return true;
  }
  return false;
}


void command(String& str, int& i){
  if (i == 0){
    movement(str);
  }
  else if (i == 1){
    if (str2bool(str)) Mouse.press(MOUSE_LEFT);
    else Mouse.release(MOUSE_LEFT);
  }
  else if (i == 2){
    if (str2bool(str)) Mouse.press(MOUSE_RIGHT);
    else Mouse.release(MOUSE_RIGHT);
  }
  else if (i == 3){
    if (str2bool(str)) Keyboard.press('q');
    else Keyboard.release('q');
  }
  else if (i == 4){
    if (str2bool(str)) Keyboard.press('w');
    else Keyboard.release('w');
  }
  else if (i == 5){
    if (str2bool(str)) Keyboard.press('e');
    else Keyboard.release('e');
  }
  else if (i == 6){
    if (str2bool(str)) Keyboard.press('r');
    else Keyboard.release('r');
  }
  else if (i == 7){
    if (str2bool(str)) Keyboard.press('a');
    else Keyboard.release('a');
  }
  else if (i == 8){
    if (str2bool(str)) Keyboard.press('s');
    else Keyboard.release('s');
  }
  else if (i == 9){
    if (str2bool(str)) Keyboard.press('d');
    else Keyboard.release('d');
  }
  else if (i == 10){
    if (str2bool(str)) Keyboard.press('f');
    else Keyboard.release('f');
  }
  else if (i == 11){
    if (str2bool(str)) Keyboard.press('g');
    else Keyboard.release('g');
  }
  else if (i == 12){
    if (str2bool(str)) Keyboard.press(KEY_LEFT_SHIFT);
    else Keyboard.release(KEY_LEFT_SHIFT);
  }
  else if (i == 13){
    if (str2bool(str)) Keyboard.press(KEY_LEFT_CTRL);
    else Keyboard.release(KEY_LEFT_CTRL);
  }
  else if (i == 14){
    if (str2bool(str)) Keyboard.press('x');
    else Keyboard.release('x');
  }
  else{
    if (str2bool(str)) Keyboard.press(' ');
    else Keyboard.release(' ');
  }  
}
/*
 * Input sequence: "mouse_x(int->str),mouse_y(int->str) mouse_hold_left mouse_hold_right hold_Q hold_W hold_E hold_R hold_A hold_S hold_D hold_F hold_G hold_Shift hold_Left_Ctrl hold_X click_Space
 */
void loop() {
  if (Serial.available() > 0){
    String msg = Serial.readStringUntil('\n');
    msg_i = 0;
    ix = 0;
    while (ix < 16){
      idx = msg.indexOf(' ');
      msgs[msg_i++] = msg.substring(0, idx);
      msg = msg.substring(idx+1);
      command(msgs[ix], ix);
      ix++;
    }
  }
}
