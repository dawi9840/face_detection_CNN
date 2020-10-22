const byte led_digital[] = {13, 12};   //r,g
byte total_digital= sizeof(led_digital);  
int8_t dir = -1;
byte i= 0;

void setup() {
  for(byte i=0; i<total_digital; i++){
    pinMode(led_digital[i], OUTPUT);
    digitalWrite(led_digital[i], LOW);
  }
  Serial.begin(9600);
}

void loop() { 
  if(Serial.available()){
    switch(Serial.read()) {
      case '0':
        digitalWrite(led_digital[0], LOW);
        Serial.println("turn off led");
        break;
      case '1':
        digitalWrite(led_digital[0], HIGH);
        Serial.println("Green led");
        break;
      case '2':
        digitalWrite(led_digital[1], HIGH);
        Serial.println("Red led");
        break;
    }
  }
}
