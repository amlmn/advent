#lang racket/base

(require racket/file)
(define freqs (map string->number (file->lines "day1.input" #:line-mode 'any)))
(define test-freqs (map string->number (file->lines "day1.input.test" #:line-mode 'any)))

; Part 1
(define (compute-sum freqs) (foldl + 0 freqs))
(displayln (compute-sum test-freqs))
(displayln (compute-sum freqs))

; Part 2
(require racket/match)
(require racket/set)

(define (find-repeat freqs)
	(define freqs-seen-so-far (mutable-set 0))
	(define (helper current-freqs current-sum)
		(match current-freqs
			[(list x xs ...) (let ([next-sum (+ current-sum x)])
				(if (set-member? freqs-seen-so-far next-sum)
					next-sum
					(begin
						(set-add! freqs-seen-so-far next-sum)
						(helper xs next-sum))))]
			[null? (helper freqs current-sum)]))
	(helper freqs 0))

(displayln (find-repeat test-freqs))
(displayln (find-repeat freqs)
