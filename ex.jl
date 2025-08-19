using Yao, YaoPlots 

YaoPlots.CircuitStyles.linecolor[] = "white" 
YaoPlots.CircuitStyles.gate_bgcolor[] = "gray" 
YaoPlots.CircuitStyles.textcolor[] = "black"

q1 = ArrayReg(bit"00")
state(q1)

q2 = ArrayReg(bit"00") + ArrayReg(bit"11") |> normalize!
state(q2)

a = (q1 |> bellcircuit)

reversebellcircuit = chain(2, control(1, 2=>X), put(1=>H))
plot(reversebellcircuit)
res = ( a |> reversebellcircuit)
state(res)

singlequbitcircuit = chain(2, put(1=>X))
plot(singlequbitcircuit)
bellstate = ArrayReg(bit"00") + ArrayReg(bit"11") |> normalize!
re = bellstate |> singlequbitcircuit
state(re)