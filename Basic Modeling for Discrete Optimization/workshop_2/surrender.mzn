% workshop 2
% select a set of trade negotiation party
enum NEGOTIATOR;
NEGOTIATOR: dummy; 

int: l; % minimum party size
int: u; % maximum party size
int: m; % minimum joint ability

array[NEGOTIATOR] of int: honor;
array[NEGOTIATOR,NEGOTIATOR] of int: joint;

var set of NEGOTIATOR: x;

constraint card(x) >= l;
constraint card(x) <= u;
constraint sum(i, j in x where i < j)(joint[i, j]) >= m;

solve maximize min([honor[i] | i in x]);

output["\(x)"];
