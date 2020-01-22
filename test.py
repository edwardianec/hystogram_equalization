import itertools

def derivative(graph):	
	derivative = []
	for i in range(1, len(graph)):
		derivative.append(graph[i] - graph[i-1])
		#print(h[i] - h[i-1])
	return derivative

def maximums(graph, maximums_count):
	maximums = {}
	previous = 0
	for i in range(1, len(graph)):
		current = graph[i] - graph[i-1]
		if (current < 0 and previous > 0): maximums[i-1] = graph[i-1]
		previous = current

	# После того, как мы нашли максимумы, мы должны упорядичить эти
	# максимумы в порядке убывания, после чего выбрать только необходимое нам количество,
	# определенное в переменной maximums_count
	maximums	= {k: v for k, v in sorted(maximums.items(), key=lambda item: item[1], reverse=True)}
	maximums 	= dict(itertools.islice(maximums.items(),maximums_count))
	maximums	= dict(sorted(maximums.items()))
	return maximums

h = [0,5,10,11,13,15,16,0,111,20,10,30,60,55,55,60,65,60,50,70,10,80,60,90,60,60,40,30,50,200,120,150]

f_deriv = derivative(h)
s_deriv = derivative(f_deriv)
maxs 	= maximums(h, 6)

print(maxs)