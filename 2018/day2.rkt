#lang racket/base

(require racket/file)
(define ids (file->lines "input.day2" #:line-mode 'any))
(define test-ids (file->lines "input.day2.test" #:line-mode 'any))

; Part 1
(require racket/function)
(define (bool->int b) (if b 1 0))
(define (make-multiset l)
	(define multiset (make-hash))
	(for ([char l])
		(hash-set! multiset char (+ 1 (hash-ref multiset char 0))))
	multiset)
(define-syntax-rule (repeat? multiset N)
	(bool->int (< 0 (length (filter (curry = N) (hash-values multiset))))))
(define (two? multiset) (repeat? multiset 2))
(define (three? multiset) (repeat? multiset 3))

(define (checksum characters)
	(define (mapper id)
		(define multiset (make-multiset (string->list id)))
		(cons (two? multiset) (three? multiset)))
	(define (folder acc bits)
		(cons (+ (car acc) (car bits)) (+ (cdr acc) (cdr bits))))
	(let* ([bits (map mapper characters)]
				 [sums (foldl folder (cons 0 0) bits)])
		(* (car sums) (cdr sums))))

(displayln (checksum test-ids))
(displayln (checksum ids))

; Part 2
(require racket/list)
(require racket/stream)
(define (make-diffs id1 id2)
	(define pairs (map cons id1 id2))
	(define indexed-pairs (map cons (stream->list (in-range (length id1))) pairs))
	(define (diff-char? ipair) (not (equal? (car (cdr ipair)) (cdr (cdr ipair)))))
	(filter diff-char? indexed-pairs))
(define (prettify id pos)
	(define (remove-at lst pos)
		(define prefix (take lst pos))
		(define suffix (drop lst (+ 1 pos)))
		(append prefix suffix))
	(list->string (remove-at id pos)))
(define (letter-diff ids)
	(define id-lists (map string->list ids))
	(define (helper id)
		(define (check? id2)
			(define diffs (make-diffs id id2))
			(if (= 1 (length diffs))
				(prettify id (caar diffs))
				#f))
		(define res (ormap check? id-lists))
		res)
	(ormap helper id-lists))

(displayln (letter-diff test-ids))
(displayln (letter-diff ids)
