% assignment 2
% select a set of raiding warrior
enum WARRIOR;
WARRIOR: dummy; 
enum CLAN; % the clan they belong too

int: l; % minimum raid party size
int: u; % maximum raid party size
int: m; % limit of hatreds

array[WARRIOR] of int: strength;
array[WARRIOR] of CLAN: clan;
array[CLAN,CLAN] of 0..1: hates;

array[1..u] of var WARRIOR: raid;
var int: obj = sum(i in raid)(strength[i]);

constraint forall(i, j in 1..u where i < j)(raid[i] != raid[j]);
constraint sum(i, j in raid where i < j)(hates[clan[i], clan[j]]) <= m;

solve maximize obj;

output ["raid = {"] ++ [if fix(raid[i]) != dummy then show(raid[i])
                        ++ if i < u /\ fix(raid[i+1]) != dummy then ", " else "" endif
                     else "" endif | i in 1..u ] ++ ["};\n"];
output["\nobj = \(obj);"];

/*
var set of WARRIOR: raid;
var int: obj = sum(i in raid)(strength[i]);

constraint card(raid) >= l;
constraint card(raid) <= u;
constraint sum(i, j in raid where i < j)(hates[clan[i], clan[j]]) <= m; 

solve maximize obj;

output["raid = \(raid diff {dummy});"];
output["\nobj = \(obj);"];
*/


