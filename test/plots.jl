L = 10
labels = rand(Bool, L);
scores = add_noise.(labels, 0.6)
curve = ROC(labels, scores)
plot(curve)

curve = PR(labels, scores)
plot(curve)

curve = DET(labels, scores)
plot(curve)
