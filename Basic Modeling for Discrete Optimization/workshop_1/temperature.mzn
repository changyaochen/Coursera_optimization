array[int] of int: readings;
int: start;

int: n = length(readings);
set of int: HOUR = 1..n;
set of int: HOUR0 = 0..n;
enum CHOICE = {heat, strongly_heat, cool, strongly_cool, do_nothing};
array[HOUR0] of var 25..30: temp;
array[HOUR] of var CHOICE: choice;
array[CHOICE] of int: adjust = [1, 4, -2, -5, 0];
array[CHOICE] of int: cost = [1, 5, 3, 9, 0];

constraint temp[0] = start;
constraint forall(h in HOUR)
                 (temp[h] = (temp[h-1] + readings[h]) div 2 + adjust[choice[h]]);
                 
var int: obj = sum(t in HOUR)(cost[choice[t]]);

solve minimize(obj);

output ["choice = \(choice)\n"];
output ["temp = \(temp);\n"];
output ["total cost is \(obj)"]
