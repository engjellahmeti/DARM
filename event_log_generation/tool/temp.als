abstract sig Activity {}
abstract sig Payload {}

abstract sig Event{
	task: one Activity,
	data: set Payload,
	tokens: set Token
}

one sig DummyPayload extends Payload {}
fact { no te:Event | DummyPayload in te.data }

one sig DummyActivity extends Activity {}

abstract sig Token {}
abstract sig SameToken extends Token {}
abstract sig DiffToken extends Token {}
lone sig DummySToken extends SameToken{}
lone sig DummyDToken extends DiffToken{}
fact { 
	no DummySToken
	no DummyDToken
	all te:Event| no (te.tokens & SameToken) or no (te.tokens & DiffToken)
}

pred True[]{some TE0}

// lang templates

pred Init(taskA: Activity) { 
	taskA = TE0.task
}

pred Existence(taskA: Activity) { 
	some te: Event | te.task = taskA
}

pred Existence(taskA: Activity, n: Int) {
	#{ te: Event | taskA = te.task } >= n
}

pred Absence(taskA: Activity) { 
	no te: Event | te.task = taskA
}

pred Absence(taskA: Activity, n: Int) {
	#{ te: Event | taskA = te.task } <= n
}

pred Exactly(taskA: Activity, n: Int) {
	#{ te: Event | taskA = te.task } = n
}

pred Choice(taskA, taskB: Activity) { 
	some te: Event | te.task = taskA or te.task = taskB
}

pred ExclusiveChoice(taskA, taskB: Activity) { 
	some te: Event | te.task = taskA or te.task = taskB
	(no te: Event | taskA = te.task) or (no te: Event | taskB = te.task )
}

pred RespondedExistence(taskA, taskB: Activity) {
	(some te: Event | taskA = te.task) implies (some ote: Event | taskB = ote.task)
}

pred Response(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and After[te, fte])
}

pred AlternateResponse(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and After[te, fte] and (no ite: Event | taskA = ite.task and After[te, ite] and After[ite, fte]))
}

pred ChainResponse(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and Next[te, fte])
}

pred Precedence(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and After[fte, te])
}

pred AlternatePrecedence(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and After[fte, te] and (no ite: Event | taskA = ite.task and After[fte, ite] and After[ite, te]))
}

pred ChainPrecedence(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (some fte: Event | taskB = fte.task and Next[fte, te])
}

pred NotRespondedExistence(taskA, taskB: Activity) {
	(some te: Event | taskA = te.task) implies (no te: Event | taskB = te.task)
}

pred NotResponse(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (no fte: Event | taskB = fte.task and After[te, fte])
}

pred NotPrecedence(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (no fte: Event | taskB = fte.task and After[fte, te])
}

pred NotChainResponse(taskA, taskB: Activity) { 
	all te: Event | taskA = te.task implies (no fte: Event | (DummyActivity = fte.task or taskB = fte.task) and Next[te, fte])
}

pred NotChainPrecedence(taskA, taskB: Activity) {
	all te: Event | taskA = te.task implies (no fte: Event | (DummyActivity = fte.task or taskB = fte.task) and Next[fte, te])
}
//-

pred example { }
run example

---------------------- end of static code block ----------------------

--------------------- generated code starts here ---------------------

one sig PayOrder extends Activity {}
one sig DeliverOrder extends Activity {}
one sig CloseOrder extends Activity {}
one sig OrderProduct extends Activity {}
one sig TE0 extends Event {}{not task=DummyActivity}
one sig TE1 extends Event {}{not task=DummyActivity}
one sig TE2 extends Event {}{not task=DummyActivity}
one sig TE3 extends Event {}{not task=DummyActivity}
one sig TE4 extends Event {}
pred Next(pre, next: Event){pre=TE0 and next=TE1 or pre=TE1 and next=TE2 or pre=TE2 and next=TE3 or pre=TE3 and next=TE4}
pred After(b, a: Event){// b=before, a=after
b=TE0 or a=TE4 or b=TE1 and not (a=TE0) or b=TE2 and (a=TE4 or a=TE3)}
fact { all te: Event | te.task = OrderProduct implies (one Price & te.data and one Amount & te.data)}
fact { all te: Event | te.task = PayOrder implies (one Resource & te.data and one Discount & te.data)}
fact { all te: Event | te.task = DeliverOrder implies (one Delivery & te.data)}
fact { all te: Event | te.task = CloseOrder implies (one PaymentType & te.data)}
fact { all te: Event | lone(Discount & te.data) }
fact { all te: Event | some (Discount & te.data) implies te.task in (PayOrder) }
fact { all te: Event | lone(Price & te.data) }
fact { all te: Event | some (Price & te.data) implies te.task in (OrderProduct) }
fact { all te: Event | lone(Amount & te.data) }
fact { all te: Event | some (Amount & te.data) implies te.task in (OrderProduct) }
fact { all te: Event | lone(Resource & te.data) }
fact { all te: Event | some (Resource & te.data) implies te.task in (PayOrder) }
fact { all te: Event | lone(PaymentType & te.data) }
fact { all te: Event | some (PaymentType & te.data) implies te.task in (CloseOrder) }
fact { all te: Event | lone(Delivery & te.data) }
fact { all te: Event | some (Delivery & te.data) implies te.task in (DeliverOrder) }
fact {
}
abstract sig Amount extends Payload {
amount: Int
}
fact { all te: Event | (lone Amount & te.data) }
pred Single(pl: Amount) {{pl.amount=1}}
fun Amount(pl: Amount): one Int {{pl.amount}}
one sig intBetween0and10001r100016 extends Amount{}{amount=15}
abstract sig PaymentType extends Payload {}
fact { all te: Event | (lone PaymentType & te.data)}
one sig Transfer extends PaymentType{}
one sig PayPal extends PaymentType{}
abstract sig Price extends Payload {
amount: Int
}
fact { all te: Event | (lone Price & te.data) }
pred Single(pl: Price) {{pl.amount=1}}
fun Amount(pl: Price): one Int {{pl.amount}}
one sig intBetween0and20001r100017 extends Price{}{amount=15}
abstract sig Discount extends Payload {
amount: Int
}
fact { all te: Event | (lone Discount & te.data) }
pred Single(pl: Discount) {{pl.amount=1}}
fun Amount(pl: Discount): one Int {{pl.amount}}
one sig intBetween0and51r100018 extends Discount{}{amount=15}
abstract sig Delivery extends Payload {}
fact { all te: Event | (lone Delivery & te.data)}
one sig Slow extends Delivery{}
one sig Fast extends Delivery{}
abstract sig Resource extends Payload {}
fact { all te: Event | (lone Resource & te.data)}
one sig Customer extends Resource{}
one sig Company extends Resource{}
pred p100019(A: Event) { { True[] } }
pred p100020(A: Event) { { True[] } }
pred p100021(A: Event) { { True[] } }
pred p100022(A: Event) { { (A.data&Delivery=Slow) } }
pred p100022c(A, B: Event) { { (B.data&PaymentType=Transfer) } }
fact { some te: Event | te.task = DeliverOrder and p100022[te]} //vc
pred p100023(A: Event) { { (A.data&Delivery=Fast) } }
pred p100023c(A, B: Event) { { (B.data&PaymentType=PayPal) } }
fact { some te: Event | te.task = DeliverOrder and p100023[te]} //vc
fact {
(not OrderProduct = TE0.task and  p100019[TE0]) or not (all te: Event | (DeliverOrder = te.task and p100023[te]) implies (some fte: Event | CloseOrder = fte.task and p100023c[te, fte] and After[te, fte])) or not (some te: Event | te.task = CloseOrder and p100021[te]) or not (some te: Event | te.task = DeliverOrder and p100020[te]) or not (all te: Event | (DeliverOrder = te.task and p100022[te]) implies (some fte: Event | CloseOrder = fte.task and p100022c[te, fte] and After[te, fte]))
}
